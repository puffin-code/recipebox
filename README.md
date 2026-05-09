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
- **Embedding cache**: Incrementally refreshes semantic search embeddings as recipes are added.
- **Ratings and notes**: Save local personal ratings and notes.
- **Checkpointed batch ingestion**: Skips completed OCR/metadata outputs and logs failures.

## Architecture

- `app.py`: Streamlit UI only: upload flow, filters, recipe cards, ratings, and recommendation display.
- `recipe_ingest.py`: OCR, metadata extraction, upload ingestion, batch processing, checkpointing, and failure logs.
- `recipe_storage.py`: Loads metadata, OCR text paths, ratings, review flags, favorites, and browser filters.
- `recipe_search.py`: Search text generation, embedding cache, ranked semantic retrieval, and OpenAI recommendation calls.
- `test_setup.py`: Older experimentation/evaluation utilities kept out of the main app path.
- `review.md`: Local examples/guidance used by metadata extraction.

Data flow:

1. Image files are OCRed into `ocr_pages/`.
2. OCR text is converted into metadata JSON in `recipe_metadata/`.
3. Metadata and ratings are loaded into a dataframe.
4. Search text is embedded and cached in `recipe_embedding_cache.pkl`.
5. Streamlit uses metadata, OCR text, ratings, and cached embeddings for browsing and recommendations.

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

The app runs locally. Use the sidebar to upload recipe images, refresh the search index, or rebuild it from scratch.

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

Batch ingestion is resumable:

- Existing OCR text is reused unless `force=True`.
- Existing metadata JSON is skipped unless `force=True`.
- OCR and metadata failures are logged and the batch continues.
- Invalid model JSON responses are saved as `.raw.txt` files for debugging.

## Search Index / Embedding Cache

RecipeBox uses cached embeddings for global cookbook recommendations. The cache is incremental: unchanged recipe pages reuse existing embeddings, while new or changed pages are embedded and saved.

Refresh the index:

```bash
python -c "from recipe_storage import load_recipe_dataframe; from recipe_search import refresh_recipe_embedding_cache; df = load_recipe_dataframe(); df = df[~df['hide_from_browse']]; print(refresh_recipe_embedding_cache(df))"
```

Rebuild from scratch:

```bash
python -c "from recipe_storage import load_recipe_dataframe; from recipe_search import refresh_recipe_embedding_cache; df = load_recipe_dataframe(); df = df[~df['hide_from_browse']]; print(refresh_recipe_embedding_cache(df, force=True))"
```

The Streamlit sidebar also has:

- **Refresh search index**: Incremental, low-cost refresh.
- **Rebuild search index from scratch**: Force-regenerates every embedding.

## Data Privacy And Ignored Files

This project intentionally keeps private/generated data out of git. The following are gitignored:

- Raw recipe images and upload folders.
- OCR text outputs.
- Metadata JSON outputs.
- Ratings and review CSVs.
- Embedding caches.
- Failure logs.
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
