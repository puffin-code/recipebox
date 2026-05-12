import argparse
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = [
    {
        "name": "original",
        "ocr_dir": "ocr_pages",
        "metadata_dir": "recipe_metadata",
        "image_dirs": ["uploaded_recipe_images", "Photos-67", "recipe_images_raw"],
    },
    {
        "name": "google",
        "ocr_dir": "ocr_pages_google",
        "metadata_dir": "recipe_metadata_google",
        "image_dirs": ["recipes_from_google_drive"],
    },
]

PLAN_COLUMNS = [
    "dataset",
    "source_image",
    "title",
    "decision_source",
    "decision",
    "status",
    "file_type",
    "source_path",
    "archive_path",
    "message",
]

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".heic", ".webp"]


def _clean(value):
    """Normalize spreadsheet values for decision checks."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_truthy_keep(value):
    """Treat common spreadsheet keep markers as protected rows."""
    return _clean(value).casefold() in {"yes", "y", "true", "1", "keep"}


def _is_keep_action(value):
    """Detect explicit keep decisions that should never be archived."""
    text = _clean(value).casefold()
    return text in {"keep", "canonical", "keep_candidate"} or text.startswith("keep:")


def _duplicate_decision(row):
    """Return the approved duplicate decision, or None if uncertain."""
    suggested_action = _clean(row.get("suggested_action", ""))
    final_action = _clean(row.get("final_action", ""))
    review_note = _clean(row.get("review_note", ""))

    if _is_truthy_keep(row.get("keep_candidate", "")):
        return None

    if any(_is_keep_action(value) for value in [suggested_action, final_action, review_note]):
        return None

    for source, value in [
        ("final_action", final_action),
        ("suggested_action", suggested_action),
        ("review_note", review_note),
    ]:
        text = value.casefold()
        if text.startswith("duplicate_of:"):
            return source, value

    final_text = final_action.casefold()
    if final_text in {"duplicate", "archive", "archive_duplicate", "hide_duplicate"}:
        return "final_action", final_action

    note_text = review_note.casefold()
    if (
        "not duplicate" not in note_text
        and "keep" not in note_text
        and ("duplicate_of:" in note_text or note_text.startswith("duplicate"))
    ):
        return "review_note", review_note

    return None


def _parse_dataset_specs(specs):
    """Parse dataset specs formatted as name:ocr_dir:metadata_dir[:image_dir,...]."""
    if not specs:
        return DEFAULT_DATASETS

    datasets = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError("Dataset specs must be name:ocr_dir:metadata_dir[:image_dir,...]")

        dataset = {
            "name": parts[0],
            "ocr_dir": parts[1],
            "metadata_dir": parts[2],
            "image_dirs": [],
        }
        if len(parts) == 4 and parts[3]:
            dataset["image_dirs"] = [item for item in parts[3].split(",") if item]
        datasets.append(dataset)

    return datasets


def _dataset_map(datasets):
    """Index dataset config by name."""
    return {dataset["name"]: dataset for dataset in datasets}


def _archive_path(archive_dir, file_type, dataset, path):
    """Build an archive path that preserves dataset origin."""
    return Path(archive_dir) / file_type / dataset / path.name


def _existing_image_paths(source_image, image_dirs):
    """Find local raw image files matching the source image when available."""
    source_path = Path(source_image)
    candidates = []

    for image_dir in image_dirs:
        directory = Path(image_dir)
        if not directory.is_dir():
            continue

        direct = directory / source_path.name
        if direct.exists():
            candidates.append(direct)
            continue

        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{source_path.stem}{suffix}"
            if candidate.exists():
                candidates.append(candidate)

    return candidates


def _candidate_files(row, dataset_config):
    """Return active files that belong to one duplicate record."""
    source_image = row["source_image"]
    stem = Path(source_image).stem
    metadata_dir = Path(dataset_config["metadata_dir"])
    ocr_dir = Path(dataset_config["ocr_dir"])

    files = [
        ("metadata", metadata_dir / f"{stem}.json"),
        ("ocr", ocr_dir / f"{stem}.txt"),
        ("logs", metadata_dir / f"{stem}.raw.txt"),
    ]

    for image_path in _existing_image_paths(source_image, dataset_config.get("image_dirs", [])):
        files.append(("images", image_path))

    return files


def _move_file(source_path, archive_path, apply):
    """Move one file only when applying; otherwise report the planned move."""
    if not source_path.exists():
        return "missing", "file not found"

    if not apply:
        return "planned", ""

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        return "skipped", "archive target already exists"

    shutil.move(str(source_path), str(archive_path))
    return "archived", ""


def build_cleanup_plan(
    duplicates="duplicate_candidates.csv",
    datasets=None,
    archive_dir="archive_duplicates",
    plan_out="cleanup_plan.csv",
    apply=False,
):
    """Plan or apply reversible duplicate cleanup moves."""
    datasets = datasets or DEFAULT_DATASETS
    configs = _dataset_map(datasets)
    duplicates_df = pd.read_csv(duplicates).fillna("")
    rows = []
    seen_records = set()
    considered = 0
    archived_records = set()
    skipped_records = set()
    missing_files = 0

    for _, duplicate_row in duplicates_df.iterrows():
        dataset = _clean(duplicate_row.get("dataset", ""))
        source_image = _clean(duplicate_row.get("source_image", ""))
        record_key = (dataset, source_image)

        if not source_image or record_key in seen_records:
            continue

        seen_records.add(record_key)
        considered += 1
        decision = _duplicate_decision(duplicate_row)
        dataset_config = configs.get(dataset)

        if decision is None:
            skipped_records.add(record_key)
            rows.append({
                "dataset": dataset,
                "source_image": source_image,
                "title": _clean(duplicate_row.get("title", "")),
                "decision_source": "",
                "decision": "",
                "status": "skipped",
                "file_type": "",
                "source_path": "",
                "archive_path": "",
                "message": "no approved duplicate decision",
            })
            continue

        decision_source, decision_text = decision

        if dataset_config is None:
            skipped_records.add(record_key)
            rows.append({
                "dataset": dataset,
                "source_image": source_image,
                "title": _clean(duplicate_row.get("title", "")),
                "decision_source": decision_source,
                "decision": decision_text,
                "status": "skipped",
                "file_type": "",
                "source_path": "",
                "archive_path": "",
                "message": "dataset is not configured",
            })
            continue

        record_had_archive = False
        for file_type, source_path in _candidate_files(duplicate_row, dataset_config):
            archive_path = _archive_path(archive_dir, file_type, dataset, source_path)
            status, message = _move_file(source_path, archive_path, apply)

            if status == "archived" or status == "planned":
                record_had_archive = True
            if status == "missing":
                missing_files += 1

            rows.append({
                "dataset": dataset,
                "source_image": source_image,
                "title": _clean(duplicate_row.get("title", "")),
                "decision_source": decision_source,
                "decision": decision_text,
                "status": status,
                "file_type": file_type,
                "source_path": str(source_path),
                "archive_path": str(archive_path),
                "message": message,
            })

        if record_had_archive:
            archived_records.add(record_key)
        else:
            skipped_records.add(record_key)

    plan_df = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    plan_path = Path(plan_out)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(plan_path, index=False)

    print("==== Duplicate Cleanup Summary ====")
    print(f"records_considered: {considered}")
    print(f"records_archived: {len(archived_records) if apply else 0}")
    print(f"records_planned: {len(archived_records) if not apply else 0}")
    print(f"records_skipped: {len(skipped_records)}")
    print(f"missing_files: {missing_files}")
    print(f"cleanup_plan: {plan_path}")
    if apply:
        print("Refresh or rebuild the search index so archived duplicates disappear from recommendations.")
    else:
        print("Dry run only. Re-run with --apply to move planned files into the archive.")

    return plan_df


def main():
    parser = argparse.ArgumentParser(
        description="Plan or apply reversible duplicate cleanup for RecipeBox."
    )
    parser.add_argument("--duplicates", default="duplicate_candidates.csv")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset spec: name:ocr_dir:metadata_dir[:image_dir,...]. Can be repeated.",
    )
    parser.add_argument("--archive_dir", default="archive_duplicates")
    parser.add_argument("--plan_out", default="cleanup_plan.csv")
    parser.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    build_cleanup_plan(
        duplicates=args.duplicates,
        datasets=_parse_dataset_specs(args.dataset),
        archive_dir=args.archive_dir,
        plan_out=args.plan_out,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
