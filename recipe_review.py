import argparse
import json
from pathlib import Path

import pandas as pd

from recipe_storage import format_user_notes

REVIEW_COLUMNS = [
    "source_image",
    "dataset",
    "title",
    "title_confidence",
    "recipe_structure",
    "likely_continuation",
    "dish_type",
    "main_ingredients",
    "short_description",
    "semantic_summary",
    "user_notes",
    "existing_review_note",
    "review_note",
    "final_action",
    "combine_target",
    "notes",
    "review_text_file",
]


def _as_list(value):
    """Normalize optional metadata list fields."""
    return value if isinstance(value, list) else []


def _join_list(value):
    """Render list fields for spreadsheet review."""
    return "; ".join(_as_list(value))


def _load_existing_review_values(*paths):
    """Read existing manual review values from matching CSVs."""
    values = {}
    preserved_columns = ["review_note", "final_action", "combine_target", "notes"]

    for path in paths:
        if path is None:
            continue

        path = Path(path)
        if not path.exists():
            continue

        df = pd.read_csv(path).fillna("")
        if "source_image" not in df.columns:
            continue

        for _, row in df.iterrows():
            source = str(row["source_image"])
            current = values.setdefault(source, {})

            for column in preserved_columns:
                if column in df.columns:
                    value = str(row[column]).strip()
                    if value:
                        current[column] = value

    return values


def _ocr_text(ocr_dir, source_image):
    """Load full OCR text for sidecar review files."""
    path = Path(ocr_dir) / f"{Path(source_image).stem}.txt"

    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")


def _write_review_text_file(ocr_review_dir, source_image, data, ocr_text):
    """Write one readable OCR review file for manual inspection."""
    path = Path(ocr_review_dir) / f"{Path(source_image).stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    content = "\n".join([
        f"source_image: {source_image}",
        f"title: {data.get('title') or ''}",
        f"recipe_structure: {data.get('recipe_structure') or 'unknown'}",
        f"likely_continuation: {data.get('likely_continuation')}",
        f"semantic_summary: {data.get('semantic_summary') or ''}",
        f"main_ingredients: {_join_list(data.get('main_ingredients'))}",
        f"user_notes: {format_user_notes(data.get('user_notes', []))}",
        "",
        "OCR TEXT",
        "========",
        ocr_text,
    ])

    path.write_text(content, encoding="utf-8")
    return str(path)


def _is_suspicious(data, title_confidence_threshold):
    """Flag metadata rows that likely need manual review."""
    title = data.get("title")
    title_confidence = data.get("title_confidence")
    recipe_structure = data.get("recipe_structure")

    try:
        confidence = float(title_confidence)
    except (TypeError, ValueError):
        confidence = None

    low_confidence = confidence is None or confidence < title_confidence_threshold

    return (
        recipe_structure != "single_complete_recipe"
        or data.get("likely_continuation") is True
        or low_confidence
        or not title
        or recipe_structure in {"multiple_recipes", "partial_recipe", "continuation_page"}
    )


def build_review_queue(
    metadata_dir="recipe_metadata",
    ocr_dir="ocr_pages",
    out="review_queue.csv",
    ocr_review_dir="review_ocr_texts",
    review_inventory=None,
    include_all=False,
    include_ocr_excerpt=False,
    overwrite=False,
    title_confidence_threshold=0.6,
    ocr_excerpt_chars=1200,
    dataset="",
):
    """Create a spreadsheet-friendly CSV of suspicious recipe pages."""
    out = Path(out)
    existing_values = _load_existing_review_values(review_inventory, None if overwrite else out)
    rows = []

    for path in sorted(Path(metadata_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        if not include_all and not _is_suspicious(data, title_confidence_threshold):
            continue

        source_image = data.get("source_image") or path.stem
        existing = existing_values.get(source_image, {})
        review_note = existing.get("review_note", "")
        ocr_text = _ocr_text(ocr_dir, source_image)
        review_text_file = _write_review_text_file(
            ocr_review_dir,
            source_image,
            data,
            ocr_text,
        )

        row = {
            "source_image": source_image,
            "dataset": dataset,
            "title": data.get("title") or "",
            "title_confidence": data.get("title_confidence"),
            "recipe_structure": data.get("recipe_structure") or "unknown",
            "likely_continuation": data.get("likely_continuation"),
            "dish_type": data.get("dish_type") or "",
            "main_ingredients": _join_list(data.get("main_ingredients")),
            "short_description": data.get("short_description") or "",
            "semantic_summary": data.get("semantic_summary") or "",
            "user_notes": format_user_notes(data.get("user_notes", [])),
            "existing_review_note": review_note,
            "review_note": review_note,
            "final_action": existing.get("final_action", ""),
            "combine_target": existing.get("combine_target", ""),
            "notes": existing.get("notes", ""),
            "review_text_file": review_text_file,
        }

        if include_ocr_excerpt:
            row["ocr_excerpt"] = ocr_text[:ocr_excerpt_chars]

        rows.append(row)

    columns = REVIEW_COLUMNS + (["ocr_excerpt"] if include_ocr_excerpt else [])
    review_df = pd.DataFrame(rows, columns=columns)
    out.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(out, index=False)

    print(f"Wrote {len(review_df)} review rows to {out}")
    return review_df


def main():
    parser = argparse.ArgumentParser(description="Generate an offline recipe review queue CSV.")
    parser.add_argument("--metadata_dir", default="recipe_metadata")
    parser.add_argument("--ocr_dir", default="ocr_pages")
    parser.add_argument("--out", default="review_queue.csv")
    parser.add_argument("--ocr_review_dir", default="review_ocr_texts")
    parser.add_argument("--review_inventory", default="recipe_review_inventory.csv")
    parser.add_argument("--include_all", action="store_true")
    parser.add_argument("--include_ocr_excerpt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--title_confidence_threshold", type=float, default=0.6)
    parser.add_argument("--ocr_excerpt_chars", type=int, default=1200)
    parser.add_argument("--dataset", default="")

    args = parser.parse_args()

    build_review_queue(
        metadata_dir=args.metadata_dir,
        ocr_dir=args.ocr_dir,
        out=args.out,
        ocr_review_dir=args.ocr_review_dir,
        review_inventory=args.review_inventory,
        include_all=args.include_all,
        include_ocr_excerpt=args.include_ocr_excerpt,
        overwrite=args.overwrite,
        title_confidence_threshold=args.title_confidence_threshold,
        ocr_excerpt_chars=args.ocr_excerpt_chars,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
