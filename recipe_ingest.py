import base64
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI()


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


def extract_recipe_metadata(recipe_text, source_image=None):
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
                    "If uncertain, use null. "
                    "Do not invent ingredients not present in the text."
                    "Return ONLY valid JSON. "
                    "You MUST include every requested field. "
                    "If uncertain, use null. "
                    "Do not invent ingredients not present in the text."
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
                            "user_notes": list of strings
                            }}
                            """,
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    metadata = json.loads(raw)

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
    metadata.setdefault("user_notes", [])

    return metadata


def ingest_uploaded_recipe(image_bytes, image_name):
    """Save an upload, OCR it, extract metadata, and persist generated files."""
    upload_dir = Path("uploaded_recipe_images")
    upload_dir.mkdir(exist_ok=True)

    image_path = upload_dir / image_name
    image_path.write_bytes(image_bytes)

    recipe_text = ocr_image(image_path)

    ocr_dir = Path("ocr_pages")
    ocr_dir.mkdir(exist_ok=True)

    txt_path = ocr_dir / f"{image_path.stem}.txt"
    txt_path.write_text(recipe_text, encoding="utf-8")

    metadata = extract_recipe_metadata(recipe_text, source_image=image_name)

    metadata_dir = Path("recipe_metadata")
    metadata_dir.mkdir(exist_ok=True)

    json_path = metadata_dir / f"{image_path.stem}.json"
    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def process_recipe_images(image_dir, limit=5):
    """Batch OCR recipe images and write OCR text plus metadata JSON."""
    image_dir = Path(image_dir)

    image_paths = sorted(image_dir.glob("*.jpg"))[:limit]

    documents = []

    for image_path in tqdm(image_paths):
        print(f"\n📸 Processing: {image_path.name}")

        recipe_text = ocr_image(image_path)

        txt_path = Path("ocr_pages") / f"{image_path.stem}.txt"
        txt_path.write_text(recipe_text, encoding="utf-8")

        metadata = extract_recipe_metadata(recipe_text, source_image=image_path.name)

        json_path = Path("recipe_metadata") / f"{image_path.stem}.json"
        json_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print(f"🏷️ Title: {metadata['title']}")

        documents.append(recipe_text)

    return documents
