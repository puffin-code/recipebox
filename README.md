# RecipeBox

RecipeBox is a private Streamlit app for digitizing, browsing, searching, and chatting with a personal recipe collection. It is built for handwritten recipes, cookbook photos, and recipe images exported from places like Google Drive.

This is a local personal tool, not a public hosted product.

## What It Does

RecipeBox turns recipe images into searchable cookbook pages:

- OCRs recipe photos with OpenAI vision models.
- Extracts structured metadata from OCR text.
- Adds semantic metadata for vague mood, season, and context queries.
- Browses recipes in a Streamlit UI.
- Searches by title, source page, ingredients, dish type, notes, and semantic fields.
- Answers questions about a single recipe.
- Recommends recipes from the whole cookbook using cached semantic embeddings.
- Uploads new recipe photos from a phone or browser.
- Stores personal ratings and notes locally.
- Supports resumable/checkpointed ingestion for large batches.

## Why It Exists

Personal recipe collections are messy: handwritten pages, phone photos, cookbook clippings, and partial pages do not behave like a tidy database. RecipeBox keeps the original OCR text and generated metadata separate, then uses semantic search to answer questions like:

- "warming recipes for a cold day"
- "uplifting spring and bright recipes for dinner in the sun"
- "impressive but not too much work"

## Features

- **OCR ingestion**: Convert recipe images into text.
- **Metadata extraction**: Title, ingredients, dish type, notes, quality flags, and structure.
- **Semantic metadata**: Mood, seasonality, texture, serving context, effort, temperature, and make-ahead potential.
- **Streamlit browser**: Filter and browse recipe cards.
- **Per-recipe assistant**: Ask prep, substitution, or cooking questions about one recipe.
- **Global recommendations**: Ask the whole cookbook for ideas and view ranked matches.
- **Create from saved recipes**: Generate new AI recipe ideas inspired by selected recipes from the collection.
- **Embedding cache**: Incrementally refreshes semantic search embeddings as recipes are added.
- **Ratings and notes**: Save local personal ratings and notes.
- **Checkpointed batch ingestion**: Skips completed OCR/metadata outputs and logs failures.

## Architecture

- `app.py`: Streamlit UI only: upload flow, filters, recipe cards, ratings, and recommendation display.
- `recipe_ingest.py`: OCR, metadata extraction, upload ingestion, batch processing, checkpointing, and failure logs.
- `recipe_storage.py`: Loads metadata, OCR text paths, ratings, review flags, favorites, and browser filters.
- `recipe_search.py`: Search text generation, embedding cache, ranked semantic retrieval, and OpenAI recommendation calls.
- `recipe_duplicates.py`: Offline duplicate candidate generator for spreadsheet review.
- `recipe_cleanup.py`: Reversible offline cleanup for approved duplicate records.
- `recipe_curation.py`: Reversible offline application of reviewed delete/combine/split decisions.
- `archive/`: Older experimentation/evaluation utilities kept out of the main app path.
- `review.md`: Local examples/guidance used by metadata extraction.

Data flow:

1. Image files are OCRed into `ocr_pages/`.
2. OCR text is converted into metadata JSON in `recipe_metadata/`.
3. Metadata and ratings are loaded into a dataframe.
4. Search text is embedded and cached in `recipe_embedding_cache.pkl`.
5. Streamlit uses metadata, OCR text, ratings, and cached embeddings for browsing, recommendations, and inspiration-based recipe creation.

The app can load multiple non-overlapping datasets at once. Configure them in `app.py` with a dataset name, OCR directory, metadata directory, and optional reviewed CSV. Each row gets a stable `record_id` like `dataset:source_image`, which keeps ratings, search results, and embedding cache entries distinct across datasets.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install streamlit openai python-dotenv pandas numpy tqdm
```

## Environment Variables

Create a local `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys.

## Running The App

```bash
streamlit run app.py
```

The app runs locally. Use the sidebar to upload recipe images, refresh the search index, or rebuild it from scratch. The **Create** tab lets you select saved recipes as inspiration, generate a clearly labeled AI recipe idea, and save it into the generated recipes dataset.

## Deployment

Pushes to `main` deploy the app to Fly.io through GitHub Actions.

## Ingesting Recipes

Run a small test batch first:

```bash
python -c "from recipe_ingest import process_recipe_images; process_recipe_images(image_dir='path/to/recipe_images', limit=5)"
```

Run a full image batch:

```bash
python -c "from recipe_ingest import process_recipe_images; process_recipe_images(image_dir='path/to/recipe_images', limit=None)"
```

Regenerate metadata from existing OCR text without rerunning OCR:

```bash
python -c "from recipe_ingest import process_ocr_pages; process_ocr_pages(ocr_dir='ocr_pages', metadata_dir='recipe_metadata', force=True)"
```

Regenerate metadata for one edited OCR file:

```bash
python -m recipe_ingest --metadata-from-ocr ocr_pages_google/combined_xxx.txt --metadata_dir recipe_metadata_google
```

Batch ingestion is resumable:

- Existing OCR text is reused unless `force=True`.
- Existing metadata JSON is skipped unless `force=True`.
- OCR and metadata failures are logged and the batch continues.
- Invalid model JSON responses are saved as `.raw.txt` files for debugging.

## Search Index / Embedding Cache

RecipeBox uses cached embeddings for global cookbook recommendations. The cache is incremental: unchanged recipe pages reuse existing embeddings, while new or changed pages are embedded and saved.

Refresh the index:

```bash
python -c "from recipe_storage import load_recipe_dataframe; from recipe_search import refresh_recipe_embedding_cache; datasets=[{'name':'original','ocr_dir':'ocr_pages','metadata_dir':'recipe_metadata','review_csv':'recipe_review_inventory.csv'},{'name':'google','ocr_dir':'ocr_pages_google','metadata_dir':'recipe_metadata_google','review_csv':'review_queue_google.csv'},{'name':'generated','ocr_dir':'generated_recipes','metadata_dir':'generated_recipe_metadata'}]; df = load_recipe_dataframe(datasets); df = df[~df['hide_from_browse']]; print(refresh_recipe_embedding_cache(df))"
```

Rebuild from scratch:

```bash
python -c "from recipe_storage import load_recipe_dataframe; from recipe_search import refresh_recipe_embedding_cache; datasets=[{'name':'original','ocr_dir':'ocr_pages','metadata_dir':'recipe_metadata','review_csv':'recipe_review_inventory.csv'},{'name':'google','ocr_dir':'ocr_pages_google','metadata_dir':'recipe_metadata_google','review_csv':'review_queue_google.csv'},{'name':'generated','ocr_dir':'generated_recipes','metadata_dir':'generated_recipe_metadata'}]; df = load_recipe_dataframe(datasets); df = df[~df['hide_from_browse']]; print(refresh_recipe_embedding_cache(df, force=True))"
```

The Streamlit sidebar also has:

- **Refresh search index**: Incremental, low-cost refresh.
- **Rebuild search index from scratch**: Force-regenerates every embedding.

## Create Tab

The Create tab is for making new AI-generated recipe ideas inspired by recipes already saved in RecipeBox. It is not a generic recipe generator.

Workflow:

1. Search saved recipes or use semantic retrieval to find inspiration candidates.
2. Add recipes to the selected inspiration set.
3. Enter ingredients on hand, ingredients to use or avoid, mood/context, dietary constraints, and time/effort preference.
4. Click **Generate AI recipe idea**.
5. Review the clearly labeled AI-generated recipe and, if useful, click **Save generated recipe**.

Saved generated recipes are written as OCR-like text files in `generated_recipes/` and metadata JSON in `generated_recipe_metadata/`. The app includes these as the `generated` dataset and refreshes the search index after saving, so they become searchable without manual cache deletion.

## Offline Review Queue

Recipe review and data curation happen outside Streamlit. Generate a compact CSV of suspicious pages plus sidecar OCR text files, edit review columns in a spreadsheet, then use the reviewed CSV as `recipe_review_inventory.csv` for the app to read.

Generate a review queue:

```bash
python -m recipe_review --metadata_dir recipe_metadata --ocr_dir ocr_pages --out review_queue.csv --ocr_review_dir review_ocr_texts
```

Generate a Google batch review queue:

```bash
python -m recipe_review --metadata_dir recipe_metadata_google --ocr_dir ocr_pages_google --out review_queue_google.csv --ocr_review_dir review_ocr_texts_google --dataset google
```

By default, the queue includes suspicious pages such as continuations, partial recipes, multiple-recipes pages, missing titles, low-confidence titles, and likely continuations. The CSV stays compact and links to `review_text_file` paths containing full OCR text for manual inspection. Existing `review_note`, `final_action`, `combine_target`, and `notes` values are preserved from `recipe_review_inventory.csv` or the output CSV when present. Use `--include_ocr_excerpt` only if you want short OCR snippets inside the spreadsheet.

The Streamlit app may read reviewed notes for filtering, but it does not write review decisions or expose review editing controls.

## Offline Duplicate Detection

Duplicate curation also happens outside Streamlit. Generate a proposal CSV, review it manually, and keep the original OCR and metadata files untouched until you decide what to do.

Run duplicate detection across the default local datasets:

```bash
python -m recipe_duplicates --out duplicate_candidates.csv
```

To specify datasets explicitly:

```bash
python -m recipe_duplicates \
  --dataset original:ocr_pages:recipe_metadata:recipe_review_inventory.csv \
  --dataset google:ocr_pages_google:recipe_metadata_google:review_queue_google.csv \
  --out duplicate_candidates.csv
```

The generated `duplicate_candidates.csv` groups highly similar pages using conservative title, ingredient, metadata/OCR text, and optional embedding-cache similarity checks. Suggested actions are only proposals: `keep`, `duplicate_of:<source_image>`, or `needs_review`. Future app behavior can read approved decisions to hide duplicates, but this command does not delete files, edit metadata JSON, or change the Streamlit UI.

## Duplicate Cleanup

After reviewing `duplicate_candidates.csv`, use the cleanup helper to move approved duplicates out of the active dataset. The cleanup is reversible: files are moved into `archive_duplicates/`, not deleted.

Dry run first:

```bash
python -m recipe_cleanup --duplicates duplicate_candidates.csv
```

Apply approved moves:

```bash
python -m recipe_cleanup --duplicates duplicate_candidates.csv --apply
```

Rows are archived only when they have an explicit duplicate decision such as `suggested_action` or `final_action` starting with `duplicate_of:`. Rows marked `keep` or `keep_candidate` are protected, and uncertain rows are skipped. The script moves matching metadata JSON, OCR text, local raw images when found, and associated `.raw.txt` debug files into `archive_duplicates/metadata/`, `archive_duplicates/ocr/`, `archive_duplicates/images/`, and `archive_duplicates/logs/`.

After applying cleanup, refresh or rebuild the search index from the Streamlit sidebar so archived duplicates disappear from recommendations.

## Reviewed Curation

When a reviewed queue has final decisions from spreadsheet review, apply them with the curation helper. It defaults to a dry run and writes `curation_plan.csv` so you can inspect every planned action before anything moves.

Dry run:

```bash
python -m recipe_curation --review_csv reviewed_queue.csv
```

Apply reviewed actions:

```bash
python -m recipe_curation --review_csv reviewed_queue.csv --apply
```

Supported actions:

- `delete`: moves active metadata, OCR text, local image files when found, and `.raw.txt` debug files into `archive_curation/`.
- `combine`: creates a new combined OCR/metadata record, then archives the original component pages only after the combined output succeeds.
- `split_page`: leaves the page active and marks it `split_pending`.
- `split_page then combine` or `split_page then delete`: leaves the page active and marks it `complex_pending`.
- blank or “No action - this recipe is complete”: keeps the page active.

For combine groups, a row whose `combine_target` contains multiple image filenames defines the group and order. Targets may use wording like “and”; the parser extracts obvious `.jpg`, `.jpeg`, and `.png` filenames. Ambiguous combine rows are marked `needs_human_review`.

The curation planner treats only primary OCR `.txt` and metadata `.json` files as required recipe records. Raw/debug files are ignored as inputs and only archived when they exist alongside an active record. If duplicate cleanup already moved a component page into `archive_duplicates/` or another archive folder, curation can still use its archived OCR/metadata as combine input while leaving that archived source in place.

After applying curation, refresh or rebuild the search index from the Streamlit sidebar.

## Data Privacy And Ignored Files

This project intentionally keeps private/generated data out of git. The following are gitignored:

- Raw recipe images and upload folders.
- OCR text outputs.
- Metadata JSON outputs.
- Ratings and review CSVs.
- Embedding caches.
- Failure logs.
- Raw model response debug files.
- `.env` and local key files.
- Streamlit and Python cache directories.

Keep actual recipe contents, photos, API keys, private file paths, and personal notes out of commits.

## Current Limitations

- Metadata quality depends on OCR quality and model output.
- Partial pages and continuation pages may still need manual review.
- Semantic recommendations are only as good as the generated metadata and OCR excerpts.
- The app is optimized for local/private use, not multi-user deployment.
- Large first-time embedding builds can take time and use API credits.

## Roadmap

- Better duplicate/continuation-page handling.
- Richer review tools for metadata correction.
- Import helpers for additional cloud/photo sources.
- Optional thumbnail previews for local browsing.
- More structured rating and meal-planning views.
- Export tools for cleaned recipe indexes.
