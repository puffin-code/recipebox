import argparse
import base64
import csv
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from recipe_config import data_path, dataset_path

load_dotenv()
client = OpenAI()


def _blank_summary():
    """Create counters for resumable batch ingestion."""
    return {
        "total_files_seen": 0,
        "ocr_generated": 0,
        "ocr_reused": 0,
        "metadata_generated": 0,
        "metadata_skipped": 0,
        "failures": 0,
    }


def _print_summary(summary):
    """Print the final batch counters."""
    print("\n==== Ingestion Summary ====")
    for key, value in summary.items():
        print(f"{key}: {value}")


def _log_failure(failure_log_path, filename, stage, error):
    """Append one failure row and keep the batch moving."""
    if failure_log_path is None:
        return

    path = Path(failure_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "stage", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "filename": filename,
            "stage": stage,
            "error": str(error),
        })


def ocr_image(image_path):
    """Extract recipe text from an uploaded image with OpenAI vision."""
    image_path = Path(image_path)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type = "image/jpeg"  # fine for .jpg/.jpeg

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract the handwritten recipe clearly. "
                            "Preserve title, ingredients, and numbered steps. "
                            "Do not summarize."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                    },
                ],
            }
        ],
    )

    return response.output_text


def extract_recipe_metadata(recipe_text, source_image=None, raw_output_path=None):
    """Extract normalized metadata JSON from OCR recipe text."""
    with open("review.md", "r", encoding="utf-8") as f:
        review_examples = f.read()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract metadata from OCR'd recipe text. "
                    "Learn from the reviewed examples below."
                    f"{review_examples}"
                    "You MUST include every requested field. "
                    "If uncertain, use null for nullable fields or unknown for enum fields. "
                    "semantic_summary and tags may be inferred from the recipe text for retrieval. "
                    "user_notes must only include notes actually written on the page. "
                    "main_ingredients must not invent ingredients not present in the OCR text. "
                    "Return ONLY valid JSON. "
                ),
            },
            {
                "role": "user",
                "content": f"""
                            Recipe text:
                            {recipe_text}

                            Return JSON with exactly these fields:

                            {{
                            "source_image": {json.dumps(source_image)},
                            "title": string or null,
                            "title_confidence": number between 0 and 1,
                            "recipe_structure": one of [
                                "single_complete_recipe",
                                "multiple_recipes",
                                "continuation_page",
                                "partial_recipe",
                                "unknown"
                            ],
                            "likely_continuation": true or false,
                            "ocr_quality_score": number between 0 and 1,
                            "dish_type": string or null,
                            "main_ingredients": list of strings,
                            "short_description": string or null,
                            "semantic_summary": one retrieval-oriented paragraph describing mood,
                                seasonality, texture, flavor profile, serving context,
                                and likely use cases, or null,
                            "vibe_tags": list of short strings such as bright, fresh,
                                warming, cozy, hearty, light, rich, spicy, comforting,
                                elegant, casual, impressive, refreshing,
                            "season_tags": list of short strings such as spring, summer,
                                autumn, winter, all-season,
                            "meal_context_tags": list of short strings such as weeknight,
                                dinner_party, lunch, side_dish, main, snack, brunch,
                                make_ahead, picnic, outdoor_meal,
                            "effort_level": one of ["low", "medium", "high", "unknown"],
                            "served_temperature": one of [
                                "hot",
                                "warm",
                                "room_temperature",
                                "cold",
                                "flexible",
                                "unknown"
                            ],
                            "make_ahead_potential": one of ["low", "medium", "high", "unknown"],
                            "user_notes": list of strings
                            }}
                            """,
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    cleaned = re.sub(r"^```json\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        metadata = json.loads(cleaned)
    except json.JSONDecodeError:
        if raw_output_path is not None:
            Path(raw_output_path).write_text(raw, encoding="utf-8")
        raise

    # Harden against model responses that omit optional fields.
    metadata.setdefault("source_image", source_image)
    metadata.setdefault("title", None)
    metadata.setdefault("title_confidence", None)
    metadata.setdefault("recipe_structure", "unknown")
    metadata.setdefault("likely_continuation", False)
    metadata.setdefault("ocr_quality_score", None)
    metadata.setdefault("dish_type", None)
    metadata.setdefault("main_ingredients", [])
    metadata.setdefault("short_description", None)
    metadata.setdefault("semantic_summary", None)
    metadata.setdefault("vibe_tags", [])
    metadata.setdefault("season_tags", [])
    metadata.setdefault("meal_context_tags", [])
    metadata.setdefault("effort_level", "unknown")
    metadata.setdefault("served_temperature", "unknown")
    metadata.setdefault("make_ahead_potential", "unknown")
    metadata.setdefault("user_notes", [])

    return metadata


def ingest_uploaded_recipe(image_bytes, image_name):
    """Save an upload, OCR it, extract metadata, and persist generated files."""
    upload_dir = data_path("uploaded_recipe_images")
    upload_dir.mkdir(exist_ok=True)

    image_path = upload_dir / image_name
    image_path.write_bytes(image_bytes)

    recipe_text = ocr_image(image_path)

    ocr_dir = data_path("ocr_pages")
    ocr_dir.mkdir(exist_ok=True)

    txt_path = ocr_dir / f"{image_path.stem}.txt"
    txt_path.write_text(recipe_text, encoding="utf-8")

    metadata = extract_recipe_metadata(recipe_text, source_image=image_name)

    metadata_dir = data_path("recipe_metadata")
    metadata_dir.mkdir(exist_ok=True)

    json_path = metadata_dir / f"{image_path.stem}.json"
    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def process_recipe_images(
    image_dir,
    ocr_dir=dataset_path("ocr_pages"),
    metadata_dir=dataset_path("recipe_metadata"),
    failure_log_path="ingestion_failures.csv",
    limit=5,
    force=False,
):
    """Batch OCR recipe images with checkpointing and failure logging."""
    image_dir = Path(image_dir)
    ocr_dir = Path(ocr_dir)
    metadata_dir = Path(metadata_dir)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    documents = []
    summary = _blank_summary()

    for image_path in tqdm(image_paths):
        summary["total_files_seen"] += 1
        print(f"\n📸 Processing: {image_path.name}")

        txt_path = ocr_dir / f"{image_path.stem}.txt"
        json_path = metadata_dir / f"{image_path.stem}.json"

        if txt_path.exists() and not force:
            recipe_text = txt_path.read_text(encoding="utf-8")
            summary["ocr_reused"] += 1
            print(f"↩️  OCR reused: {txt_path}")
        else:
            try:
                recipe_text = ocr_image(image_path)
                txt_path.write_text(recipe_text, encoding="utf-8")
                summary["ocr_generated"] += 1
                print(f"✅ OCR generated: {txt_path}")
            except Exception as exc:
                summary["failures"] += 1
                _log_failure(failure_log_path, image_path.name, "ocr", exc)
                print(f"❌ OCR failed: {image_path.name} | {exc}")
                continue

        if json_path.exists() and not force:
            summary["metadata_skipped"] += 1
            print(f"⏭️  Metadata skipped: {json_path}")
            documents.append(recipe_text)
            continue

        try:
            metadata = extract_recipe_metadata(
                recipe_text,
                source_image=image_path.name,
                raw_output_path=metadata_dir / f"{image_path.stem}.raw.txt",
            )
            json_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            summary["metadata_generated"] += 1
            print(f"✅ Metadata generated: {json_path}")
            print(f"🏷️ Title: {metadata['title']}")
        except Exception as exc:
            summary["failures"] += 1
            _log_failure(failure_log_path, image_path.name, "metadata", exc)
            print(f"❌ Metadata failed: {image_path.name} | {exc}")
            continue

        documents.append(recipe_text)

    _print_summary(summary)
    return documents


def process_ocr_pages(
    ocr_dir=dataset_path("ocr_pages"),
    metadata_dir=dataset_path("recipe_metadata"),
    failure_log_path="ingestion_failures.csv",
    limit=None,
    force=False,
):
    """Generate metadata from existing OCR text files with checkpointing."""
    ocr_paths = sorted(Path(ocr_dir).glob("*.txt"))

    if limit is not None:
        ocr_paths = ocr_paths[:limit]

    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    summary = _blank_summary()

    for ocr_path in tqdm(ocr_paths):
        summary["total_files_seen"] += 1
        print(f"\n📝 Processing OCR: {ocr_path.name}")
        summary["ocr_reused"] += 1

        try:
            recipe_text = ocr_path.read_text(encoding="utf-8")
        except Exception as exc:
            summary["failures"] += 1
            _log_failure(failure_log_path, ocr_path.name, "ocr", exc)
            print(f"❌ OCR text read failed: {ocr_path.name} | {exc}")
            continue

        json_path = metadata_dir / f"{ocr_path.stem}.json"

        if json_path.exists() and not force:
            summary["metadata_skipped"] += 1
            print(f"⏭️  Metadata skipped: {json_path}")
            documents.append(recipe_text)
            continue

        source_image = ocr_path.stem
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
                source_image = existing.get("source_image") or source_image
            except json.JSONDecodeError:
                pass

        try:
            metadata = extract_recipe_metadata(
                recipe_text,
                source_image=source_image,
                raw_output_path=metadata_dir / f"{ocr_path.stem}.raw.txt",
            )
            json_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            summary["metadata_generated"] += 1
            print(f"✅ Metadata generated: {json_path}")
            print(f"🏷️ Title: {metadata['title']}")
        except Exception as exc:
            summary["failures"] += 1
            _log_failure(failure_log_path, ocr_path.name, "metadata", exc)
            print(f"❌ Metadata failed: {ocr_path.name} | {exc}")
            continue

        documents.append(recipe_text)

    _print_summary(summary)
    return documents


def metadata_from_ocr_file(
    ocr_path,
    metadata_dir=dataset_path("recipe_metadata"),
    source_image=None,
):
    """Regenerate metadata JSON for one OCR text file."""
    ocr_path = Path(ocr_path)
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    recipe_text = ocr_path.read_text(encoding="utf-8")
    source_image = source_image or ocr_path.with_suffix(".jpg").name
    json_path = metadata_dir / f"{ocr_path.stem}.json"

    metadata = extract_recipe_metadata(
        recipe_text,
        source_image=source_image,
        raw_output_path=metadata_dir / f"{ocr_path.stem}.raw.txt",
    )
    metadata["source_image"] = source_image
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json_path)
    return json_path


def main():
    parser = argparse.ArgumentParser(description="RecipeBox ingestion utilities.")
    parser.add_argument(
        "--metadata-from-ocr",
        help="Regenerate metadata JSON for one OCR .txt file.",
    )
    parser.add_argument("--metadata_dir", default="recipe_metadata")
    parser.add_argument(
        "--source_image",
        help="Optional source_image value for metadata; defaults to OCR filename with .jpg.",
    )

    args = parser.parse_args()

    if args.metadata_from_ocr:
        metadata_from_ocr_file(
            args.metadata_from_ocr,
            metadata_dir=args.metadata_dir,
            source_image=args.source_image,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
