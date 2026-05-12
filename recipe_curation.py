import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = [
    {
        "name": "binder",
        "ocr_dir": "ocr_pages",
        "metadata_dir": "recipe_metadata",
        "image_dirs": ["uploaded_recipe_images", "Photos-67", "recipe_images_raw"],
    },
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
    "source_image",
    "dataset",
    "action",
    "status",
    "source_location",
    "active_ocr_found",
    "active_metadata_found",
    "archive_ocr_found",
    "archive_metadata_found",
    "reason",
    "title",
    "output_source_image",
    "active_ocr_path",
    "active_metadata_path",
    "archive_ocr_path",
    "archive_metadata_path",
    "archive_path",
]

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".heic", ".webp"]
FILENAME_PATTERN = re.compile(r"[\w .()\-]+?\.(?:jpe?g|png)", re.IGNORECASE)
PARTIAL_HINT_PATTERN = re.compile(r"\b(top|bottom|half|left|right|section|part)\b", re.IGNORECASE)


def _clean(value):
    """Normalize spreadsheet cells."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _stem(source_image):
    """Return the canonical filename stem for OCR/metadata lookup."""
    return Path(_clean(source_image)).stem


def _slug(value):
    """Create a filesystem-safe label for combined records."""
    text = re.sub(r"[^a-z0-9]+", "_", _clean(value).casefold()).strip("_")
    return text[:48] or "recipe"


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
    """Index configured datasets by name."""
    return {dataset["name"]: dataset for dataset in datasets}


def _archive_roots(archive_dirs):
    """Return existing archive roots plus any archive-like repo directories."""
    roots = []
    for value in archive_dirs:
        path = Path(value)
        if path.exists():
            roots.append(path)

    for path in Path(".").glob("archive*"):
        if path.is_dir() and path not in roots:
            roots.append(path)

    return roots


def _primary_paths(config, source_image):
    """Resolve active primary OCR and metadata paths for one record."""
    stem = _stem(source_image)
    return {
        "active_ocr": Path(config["ocr_dir"]) / f"{stem}.txt",
        "active_metadata": Path(config["metadata_dir"]) / f"{stem}.json",
    }


def _find_archived_primary(source_image, dataset, file_type, archive_roots):
    """Find archived OCR or metadata primary files for a reviewed page."""
    stem = _stem(source_image)
    suffix = ".txt" if file_type == "ocr" else ".json"

    for root in archive_roots:
        preferred = root / file_type / dataset / f"{stem}{suffix}"
        if dataset and preferred.exists():
            return preferred
        if dataset:
            continue

        for path in root.glob(f"{file_type}/**/{stem}{suffix}"):
            if path.name.endswith(".raw.txt"):
                continue
            return path

    return None


def _source_location(active_ocr, active_metadata, archive_ocr, archive_metadata):
    """Describe where primary record files were found."""
    active = active_ocr or active_metadata
    archived = archive_ocr or archive_metadata
    if active and archived:
        return "active_and_archive"
    if active:
        return "active"
    if archived:
        return "archive"
    return "missing"


def _resolve_record(config, dataset, source_image, archive_roots):
    """Resolve only primary recipe files, excluding raw/debug/log artifacts."""
    paths = _primary_paths(config, source_image)
    active_ocr = paths["active_ocr"] if paths["active_ocr"].exists() else None
    active_metadata = paths["active_metadata"] if paths["active_metadata"].exists() else None
    archive_ocr = _find_archived_primary(source_image, dataset, "ocr", archive_roots)
    archive_metadata = _find_archived_primary(source_image, dataset, "metadata", archive_roots)

    return {
        "active_ocr": active_ocr,
        "active_metadata": active_metadata,
        "archive_ocr": archive_ocr,
        "archive_metadata": archive_metadata,
        "source_location": _source_location(
            active_ocr,
            active_metadata,
            archive_ocr,
            archive_metadata,
        ),
    }


def _resolve_dataset_config(configs, dataset, source_image, archive_roots):
    """Resolve explicit dataset names, falling back to primary file presence."""
    if dataset in configs:
        return dataset, configs[dataset]

    matches = []
    for name, config in configs.items():
        record = _resolve_record(config, name, source_image, archive_roots)
        if record["source_location"] != "missing":
            matches.append((name, config))

    if len(matches) == 1:
        return matches[0]
    if matches:
        unique_locations = {
            (
                str(Path(config["ocr_dir"])),
                str(Path(config["metadata_dir"])),
            )
            for _, config in matches
        }
        if len(unique_locations) == 1:
            return matches[0]
    return dataset, None


def _is_non_primary_source(source_image):
    """Skip rows that refer to generated artifacts rather than recipe pages."""
    source = _clean(source_image).casefold()
    name = Path(source).name
    return (
        not source
        or name.endswith(".raw.txt")
        or name.endswith(".csv")
        or "ingestion_failure" in name
        or "curation_plan" in name
        or "duplicate_candidate" in name
        or "review_queue" in name
        or source in {"logs", "log"}
    )


def _existing_image_paths(source_image, image_dirs):
    """Find local raw image files when available."""
    source_path = Path(source_image)
    paths = []
    for image_dir in image_dirs:
        directory = Path(image_dir)
        if not directory.is_dir():
            continue

        direct = directory / source_path.name
        if direct.exists():
            paths.append(direct)
            continue

        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{source_path.stem}{suffix}"
            if candidate.exists():
                paths.append(candidate)
    return paths


def _associated_active_files(config, source_image):
    """Return optional associated files to archive, without making them required."""
    stem = _stem(source_image)
    files = []
    raw_path = Path(config["metadata_dir"]) / f"{stem}.raw.txt"
    if raw_path.exists():
        files.append(("logs", raw_path))

    for image_path in _existing_image_paths(source_image, config.get("image_dirs", [])):
        files.append(("images", image_path))
    return files


def _archive_path(archive_dir, file_type, dataset, source_path):
    """Build a reversible archive destination for active files."""
    return Path(archive_dir) / file_type / dataset / source_path.name


def _move_to_archive(source_path, archive_path, apply):
    """Plan or move an active file into the reversible archive."""
    if not source_path.exists():
        return "missing", "file not found"
    if not apply:
        return "archived_planned", ""

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        return "already_archived", "archive target already exists"

    shutil.move(str(source_path), str(archive_path))
    return "archived", ""


def _parse_filenames(value):
    """Extract obvious image filenames from free-form combine targets."""
    filenames = []
    seen = set()
    text = re.sub(r"\([^)]*\)", " ", _clean(value))
    text = re.sub(r"\b(and|with|plus)\b", ",", text, flags=re.IGNORECASE)

    for match in FILENAME_PATTERN.findall(text):
        filename = re.sub(r"\s+", " ", match).strip(" ,;")
        filename = re.sub(r"^(and|with|plus)\s+", "", filename, flags=re.IGNORECASE)
        key = filename.casefold()
        if key not in seen:
            filenames.append(filename)
            seen.add(key)
    return filenames


def _normalized_action(row):
    """Classify reviewed curation action conservatively."""
    text = _clean(row.get("final_action", "")).casefold()

    if not text or "no action" in text or "complete recipe" in text:
        return "keep"

    has_split = "split_page" in text or "split page" in text or "split" in text
    has_combine = "combine" in text
    has_delete = "delete" in text

    if has_split and (has_combine or has_delete):
        return "complex_pending"
    if has_split:
        return "split_pending"
    if has_delete:
        return "delete"
    if has_combine:
        return "combine"
    return "needs_human_review"


def _partial_hint(row):
    """Detect free-form notes that imply partial-page work is needed."""
    text = " ".join([
        _clean(row.get("combine_target", "")),
        _clean(row.get("notes", "")),
    ])
    return bool(PARTIAL_HINT_PATTERN.search(text))


def _review_rows_by_source(review_df):
    """Index review rows by dataset/source for group lookup."""
    rows = {}
    for _, row in review_df.iterrows():
        dataset = _clean(row.get("dataset", ""))
        source = _clean(row.get("source_image", ""))
        if source and not _is_non_primary_source(source):
            rows[(dataset, source)] = row
    return rows


def _build_combine_groups(review_df):
    """Find rows whose combine_target defines a multi-page group."""
    groups = []
    seen_keys = set()

    for _, row in review_df.iterrows():
        if _normalized_action(row) != "combine":
            continue

        dataset = _clean(row.get("dataset", ""))
        source = _clean(row.get("source_image", ""))
        if _is_non_primary_source(source):
            continue
        filenames = _parse_filenames(row.get("combine_target", ""))

        if len(filenames) >= 2:
            members = filenames
        elif len(filenames) == 1 and source and filenames[0] != source:
            members = [source, filenames[0]]
        else:
            continue

        if source and source not in members:
            members = [source] + members

        key = (dataset, tuple(filename.casefold() for filename in members))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        groups.append({
            "dataset": dataset,
            "source_image": source,
            "members": members,
            "title": _clean(row.get("title", "")),
            "complex_pending": _partial_hint(row),
        })

    return groups


def _combined_source_name(group):
    """Create a stable source name for combined OCR/metadata outputs."""
    title = group["title"] or Path(group["members"][0]).stem
    digest = hashlib.sha1("|".join(group["members"]).encode("utf-8")).hexdigest()[:10]
    return f"combined_{_slug(title)}_{digest}.jpg"


def _combined_output_paths(config, output_source):
    """Return active output OCR/metadata paths for a combined recipe."""
    stem = _stem(output_source)
    return (
        Path(config["ocr_dir"]) / f"{stem}.txt",
        Path(config["metadata_dir"]) / f"{stem}.json",
    )


def _record_plan_base(row, dataset, source_image, action, status, record, reason="", **extra):
    """Build one row for curation_plan.csv."""
    return {
        "source_image": source_image,
        "dataset": dataset,
        "action": action,
        "status": status,
        "source_location": record.get("source_location", ""),
        "active_ocr_found": bool(record.get("active_ocr")),
        "active_metadata_found": bool(record.get("active_metadata")),
        "archive_ocr_found": bool(record.get("archive_ocr")),
        "archive_metadata_found": bool(record.get("archive_metadata")),
        "reason": reason,
        "title": _clean(row.get("title", "")),
        "output_source_image": extra.get("output_source_image", ""),
        "active_ocr_path": str(record.get("active_ocr") or ""),
        "active_metadata_path": str(record.get("active_metadata") or ""),
        "archive_ocr_path": str(record.get("archive_ocr") or ""),
        "archive_metadata_path": str(record.get("archive_metadata") or ""),
        "archive_path": extra.get("archive_path", ""),
    }


def _read_ocr_for_combine(record):
    """Read OCR from active first, then archive fallback."""
    path = record.get("active_ocr") or record.get("archive_ocr")
    if not path:
        return None
    return path.read_text(encoding="utf-8")


def _write_combined_outputs(group, config, member_records, apply):
    """Write combined OCR and metadata before any active source page is archived."""
    output_source = _combined_source_name(group)
    output_ocr_path, output_metadata_path = _combined_output_paths(config, output_source)
    sections = []
    missing = []

    for source_image, record in member_records:
        text = _read_ocr_for_combine(record)
        if text is None:
            missing.append(source_image)
            continue
        sections.append(f"===== PAGE: {source_image} =====\n{text}")

    if missing:
        return output_source, "skipped_missing", f"missing OCR for: {', '.join(missing)}"

    if output_ocr_path.exists() or output_metadata_path.exists():
        return output_source, "already_exists", "combined output already exists"

    if not apply:
        return output_source, "combined_planned", ""

    ocr_text = "\n\n".join(sections)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from recipe_ingest import extract_recipe_metadata

        metadata = extract_recipe_metadata(
            ocr_text,
            source_image=output_source,
            raw_output_path=output_metadata_path.with_suffix(".raw.txt"),
        )
    except Exception as exc:
        return output_source, "needs_human_review", f"metadata generation failed: {exc}"

    output_ocr_path.parent.mkdir(parents=True, exist_ok=True)
    output_ocr_path.write_text(ocr_text, encoding="utf-8")
    metadata["source_image"] = output_source
    output_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_source, "combined_created", ""


def _archive_active_record_files(config, dataset, source_image, archive_dir, apply):
    """Archive active primary files and optional associated artifacts."""
    planned = []
    for file_type, path in [
        ("metadata", Path(config["metadata_dir"]) / f"{_stem(source_image)}.json"),
        ("ocr", Path(config["ocr_dir"]) / f"{_stem(source_image)}.txt"),
    ]:
        if path.exists():
            archive_path = _archive_path(archive_dir, file_type, dataset, path)
            status, reason = _move_to_archive(path, archive_path, apply)
            planned.append((file_type, path, archive_path, status, reason))

    for file_type, path in _associated_active_files(config, source_image):
        archive_path = _archive_path(archive_dir, file_type, dataset, path)
        status, reason = _move_to_archive(path, archive_path, apply)
        planned.append((file_type, path, archive_path, status, reason))

    return planned


def build_curation_plan(
    review_csv="reviewed_queue.csv",
    datasets=None,
    archive_dir="archive_curation",
    archive_search_dirs=None,
    plan_out="curation_plan.csv",
    apply=False,
):
    """Plan or apply reviewed offline curation decisions."""
    datasets = datasets or DEFAULT_DATASETS
    configs = _dataset_map(datasets)
    archive_roots = _archive_roots(archive_search_dirs or ["archive_duplicates", archive_dir])
    review_df = pd.read_csv(review_csv).fillna("")
    rows_by_source = _review_rows_by_source(review_df)
    combine_groups = _build_combine_groups(review_df)
    combined_members = {
        (_clean(group["dataset"]), member)
        for group in combine_groups
        for member in group["members"]
    }
    plan_rows = []
    summary = {
        "kept": 0,
        "already_archived": 0,
        "archived_planned": 0,
        "combined_planned": 0,
        "split_pending": 0,
        "complex_pending": 0,
        "skipped_missing": 0,
        "needs_human_review": 0,
    }

    for group in combine_groups:
        marker_row = rows_by_source.get((group["dataset"], group["source_image"]), pd.Series(group))
        dataset, config = _resolve_dataset_config(
            configs,
            group["dataset"],
            group["members"][0],
            archive_roots,
        )

        if group["complex_pending"]:
            summary["complex_pending"] += len(group["members"])
            plan_rows.append(_record_plan_base(
                marker_row,
                dataset,
                group["source_image"],
                "combine",
                "complex_pending",
                {},
                "combine target/notes mention partial-page work",
            ))
            continue

        if config is None:
            summary["needs_human_review"] += len(group["members"])
            plan_rows.append(_record_plan_base(
                marker_row,
                dataset,
                group["source_image"],
                "combine",
                "needs_human_review",
                {},
                "dataset is not configured or ambiguous",
            ))
            continue

        member_records = []
        for source_image in group["members"]:
            record = _resolve_record(config, dataset, source_image, archive_roots)
            member_records.append((source_image, record))

        output_source, combine_status, reason = _write_combined_outputs(
            group,
            config,
            member_records,
            apply,
        )

        if combine_status == "combined_planned":
            summary["combined_planned"] += 1
        elif combine_status == "combined_created":
            summary["combined_planned"] += 1
        elif combine_status == "skipped_missing":
            summary["skipped_missing"] += 1
        elif combine_status == "needs_human_review":
            summary["needs_human_review"] += 1

        plan_rows.append(_record_plan_base(
            marker_row,
            dataset,
            group["source_image"],
            "combine_create",
            combine_status,
            {},
            reason,
            output_source_image=output_source,
        ))

        if combine_status not in {"combined_planned", "combined_created", "already_exists"}:
            continue

        for source_image, record in member_records:
            component_row = rows_by_source.get((group["dataset"], source_image), marker_row)

            if record["source_location"] == "archive":
                summary["already_archived"] += 1
                plan_rows.append(_record_plan_base(
                    component_row,
                    dataset,
                    source_image,
                    "combine_component",
                    "already_archived",
                    record,
                    "component is already archived; using archived OCR as input",
                    output_source_image=output_source,
                ))
                continue

            if not record["active_ocr"] and not record["active_metadata"]:
                summary["skipped_missing"] += 1
                plan_rows.append(_record_plan_base(
                    component_row,
                    dataset,
                    source_image,
                    "combine_component",
                    "skipped_missing",
                    record,
                    "no active or archived primary files found",
                    output_source_image=output_source,
                ))
                continue

            moves = _archive_active_record_files(config, dataset, source_image, archive_dir, apply)
            if moves:
                summary["archived_planned"] += 1
                archive_paths = "; ".join(str(item[2]) for item in moves)
                status = "archived" if apply else "archived_planned"
                reason_text = "active component files archived" if apply else "active component files would be archived"
            else:
                archive_paths = ""
                status = "already_archived"
                reason_text = "no active primary files remain"
                summary["already_archived"] += 1

            plan_rows.append(_record_plan_base(
                component_row,
                dataset,
                source_image,
                "combine_component",
                status,
                record,
                reason_text,
                output_source_image=output_source,
                archive_path=archive_paths,
            ))

    processed = set(combined_members)

    for _, row in review_df.iterrows():
        dataset = _clean(row.get("dataset", ""))
        source_image = _clean(row.get("source_image", ""))
        if _is_non_primary_source(source_image) or (dataset, source_image) in processed:
            continue

        action = _normalized_action(row)
        resolved_dataset, config = _resolve_dataset_config(configs, dataset, source_image, archive_roots)
        record = (
            _resolve_record(config, resolved_dataset, source_image, archive_roots)
            if config is not None
            else {}
        )

        if action == "keep":
            summary["kept"] += 1
            plan_rows.append(_record_plan_base(
                row,
                resolved_dataset,
                source_image,
                "keep",
                "kept",
                record,
                "no curation action",
            ))
            continue

        if action == "split_pending":
            summary["split_pending"] += 1
            plan_rows.append(_record_plan_base(
                row,
                resolved_dataset,
                source_image,
                "split_page",
                "split_pending",
                record,
                "automatic page splitting is not implemented",
            ))
            continue

        if action == "complex_pending":
            summary["complex_pending"] += 1
            plan_rows.append(_record_plan_base(
                row,
                resolved_dataset,
                source_image,
                "complex_split",
                "complex_pending",
                record,
                "split_page combined with combine/delete needs manual handling",
            ))
            continue

        if action == "combine":
            summary["needs_human_review"] += 1
            plan_rows.append(_record_plan_base(
                row,
                resolved_dataset,
                source_image,
                "combine",
                "needs_human_review",
                record,
                "combine action has no unambiguous filename group",
            ))
            continue

        if action == "needs_human_review" or config is None:
            summary["needs_human_review"] += 1
            plan_rows.append(_record_plan_base(
                row,
                resolved_dataset,
                source_image,
                action,
                "needs_human_review",
                record,
                "unrecognized action or dataset",
            ))
            continue

        if action == "delete":
            if record.get("source_location") == "archive":
                summary["already_archived"] += 1
                plan_rows.append(_record_plan_base(
                    row,
                    resolved_dataset,
                    source_image,
                    "delete",
                    "already_archived",
                    record,
                    "primary files are already archived",
                ))
                continue

            if record.get("source_location") == "missing":
                summary["skipped_missing"] += 1
                plan_rows.append(_record_plan_base(
                    row,
                    resolved_dataset,
                    source_image,
                    "delete",
                    "skipped_missing",
                    record,
                    "no active or archived primary files found",
                ))
                continue

            moves = _archive_active_record_files(config, resolved_dataset, source_image, archive_dir, apply)
            archive_paths = "; ".join(str(item[2]) for item in moves)
            summary["archived_planned"] += 1
            plan_rows.append(_record_plan_base(
                row,
                resolved_dataset,
                source_image,
                "delete",
                "archived" if apply else "archived_planned",
                record,
                "active primary files archived" if apply else "active primary files would be archived",
                archive_path=archive_paths,
            ))

    plan_df = pd.DataFrame(plan_rows, columns=PLAN_COLUMNS)
    plan_path = Path(plan_out)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(plan_path, index=False)

    print("==== Curation Summary ====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"curation_plan: {plan_path}")
    if apply:
        print("Refresh or rebuild the search index so curated changes appear in recommendations.")
    else:
        print("Dry run only. Re-run with --apply after reviewing the plan.")

    return plan_df


def main():
    parser = argparse.ArgumentParser(
        description="Plan or apply reviewed RecipeBox curation decisions."
    )
    parser.add_argument("--review_csv", default="reviewed_queue.csv")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset spec: name:ocr_dir:metadata_dir[:image_dir,...]. Can be repeated.",
    )
    parser.add_argument("--archive_dir", default="archive_curation")
    parser.add_argument(
        "--archive_search_dir",
        action="append",
        help="Archive root to search for fallback OCR/metadata. Can be repeated.",
    )
    parser.add_argument("--plan_out", default="curation_plan.csv")
    parser.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    build_curation_plan(
        review_csv=args.review_csv,
        datasets=_parse_dataset_specs(args.dataset),
        archive_dir=args.archive_dir,
        archive_search_dirs=args.archive_search_dir,
        plan_out=args.plan_out,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
