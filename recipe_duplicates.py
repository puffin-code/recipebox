import argparse
import math
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import numpy as np

from recipe_storage import format_user_notes, load_ocr_text_for_source, load_recipe_dataframe

DEFAULT_DATASETS = [
    {
        "name": "original",
        "ocr_dir": "ocr_pages",
        "metadata_dir": "recipe_metadata",
        "review_csv": "recipe_review_inventory.csv",
    },
    {
        "name": "google",
        "ocr_dir": "ocr_pages_google",
        "metadata_dir": "recipe_metadata_google",
        "review_csv": "review_queue_google.csv",
    },
]

DUPLICATE_COLUMNS = [
    "duplicate_group_id",
    "source_image",
    "dataset",
    "title",
    "title_confidence",
    "main_ingredients",
    "short_description",
    "semantic_summary",
    "duplicate_reason",
    "similarity_score",
    "suggested_action",
    "keep_candidate",
    "review_note",
]

UNTITLED_VALUES = {"", "untitled", "untitled recipe", "untitled recipe/page"}


def _normalize_text(value):
    """Normalize noisy recipe text for conservative duplicate comparison."""
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _as_list(value):
    """Return list-like dataframe fields as a clean list of strings."""
    return [str(item).strip() for item in value] if isinstance(value, list) else []


def _join_list(value):
    """Render list metadata for the review CSV."""
    return "; ".join(_as_list(value))


def _title_is_useful(title):
    """Avoid grouping pages only because they share a generic title."""
    return _normalize_text(title) not in UNTITLED_VALUES


def _jaccard(left, right):
    """Compute ingredient token overlap."""
    left_set = {_normalize_text(item) for item in left if _normalize_text(item)}
    right_set = {_normalize_text(item) for item in right if _normalize_text(item)}

    if not left_set or not right_set:
        return 0.0

    return len(left_set & right_set) / len(left_set | right_set)


def _cosine(left, right):
    """Compare cached embeddings without importing the search/OpenAI path."""
    if left is None or right is None:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if not left_norm or not right_norm:
        return 0.0

    return dot / (left_norm * right_norm)


def _load_embedding_cache(cache_path):
    """Load optional existing recipe embeddings for semantic duplicate checks."""
    path = Path(cache_path)
    if not path.exists():
        return {}

    try:
        with path.open("rb") as f:
            cache = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return {}

    embeddings = {}
    for record in cache.get("records", []):
        key = record.get("record_id") or record.get("source_image")
        embedding = record.get("embedding")
        if key and embedding is not None:
            embeddings[key] = embedding

    return embeddings


def build_comparison_text(row, ocr_excerpt_chars=1200):
    """Build compact text used for fuzzy duplicate comparison."""
    ocr_text = load_ocr_text_for_source(
        row["source_image"],
        row.get("ocr_dir", "ocr_pages"),
    )
    parts = [
        row.get("title", ""),
        " ".join(_as_list(row.get("main_ingredients", []))),
        row.get("short_description", ""),
        row.get("semantic_summary", ""),
        row.get("source_image", ""),
        ocr_text[:ocr_excerpt_chars],
    ]
    return _normalize_text(" ".join(str(part) for part in parts if part))


def _build_records(df, embeddings, ocr_excerpt_chars):
    """Normalize dataframe rows once before pairwise comparison."""
    records = []
    for _, row in df.iterrows():
        record_id = row.get("record_id", row["source_image"])
        records.append({
            "record_id": record_id,
            "dataset": row.get("dataset", ""),
            "source_image": row["source_image"],
            "title": row.get("title", ""),
            "normalized_title": _normalize_text(row.get("title", "")),
            "title_confidence": row.get("title_confidence", ""),
            "main_ingredients": _as_list(row.get("main_ingredients", [])),
            "short_description": row.get("short_description", ""),
            "semantic_summary": row.get("semantic_summary", ""),
            "user_notes": format_user_notes(row.get("user_notes", [])),
            "comparison_text": build_comparison_text(row, ocr_excerpt_chars),
            "embedding": embeddings.get(record_id),
        })
    return records


def _pair_duplicate_signal(left, right):
    """Return a conservative duplicate signal for one pair, or None."""
    title_ratio = SequenceMatcher(
        None,
        left["normalized_title"],
        right["normalized_title"],
    ).ratio()
    ingredient_overlap = _jaccard(left["main_ingredients"], right["main_ingredients"])
    same_title = (
        _title_is_useful(left["title"])
        and left["normalized_title"] == right["normalized_title"]
    )
    both_titles_useful = _title_is_useful(left["title"]) and _title_is_useful(right["title"])

    plausible_pair = (
        same_title
        or (both_titles_useful and title_ratio >= 0.75)
        or ingredient_overlap >= 0.30
    )
    if not plausible_pair:
        return None

    # Full metadata/OCR fuzzy matching is the expensive part, so only run it
    # for pairs that already have a strong metadata signal.
    check_text = same_title or (both_titles_useful and title_ratio >= 0.75) or ingredient_overlap >= 0.50
    text_ratio = (
        SequenceMatcher(
            None,
            left["comparison_text"],
            right["comparison_text"],
        ).ratio()
        if check_text
        else 0.0
    )
    embedding_score = _cosine(left.get("embedding"), right.get("embedding"))

    reasons = []
    score = max(title_ratio, ingredient_overlap, text_ratio, embedding_score)

    if same_title and (ingredient_overlap >= 0.35 or text_ratio >= 0.78 or embedding_score >= 0.90):
        reasons.append("exact_title_match")

    if (
        _title_is_useful(left["title"])
        and _title_is_useful(right["title"])
        and title_ratio >= 0.92
        and (ingredient_overlap >= 0.30 or text_ratio >= 0.82 or embedding_score >= 0.90)
    ):
        reasons.append("near_title_match")

    if (
        ingredient_overlap >= 0.80
        and min(len(left["main_ingredients"]), len(right["main_ingredients"])) >= 3
        and (title_ratio >= 0.72 or text_ratio >= 0.82 or embedding_score >= 0.90)
    ):
        reasons.append("high_ingredient_overlap")

    if embedding_score >= 0.96 and (title_ratio >= 0.60 or ingredient_overlap >= 0.35):
        reasons.append("high_embedding_similarity")
        score = max(score, embedding_score)

    if text_ratio >= 0.92 and (title_ratio >= 0.70 or ingredient_overlap >= 0.35):
        reasons.append("high_text_similarity")

    if not reasons:
        return None

    details = (
        f"{', '.join(sorted(set(reasons)))}; "
        f"title={title_ratio:.3f}; ingredients={ingredient_overlap:.3f}; "
        f"text={text_ratio:.3f}; embedding={embedding_score:.3f}"
    )
    return score, details


def _embedding_edges(records, threshold=0.96):
    """Find highly similar cached embeddings with a vectorized local pass."""
    embedded = [
        record for record in records
        if record.get("embedding") is not None
    ]
    if len(embedded) < 2:
        return []

    matrix = np.array([record["embedding"] for record in embedded], dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 0
    if not valid.any():
        return []

    matrix = matrix[valid] / norms[valid, None]
    embedded = [record for record, is_valid in zip(embedded, valid) if is_valid]
    scores = matrix @ matrix.T
    edges = []

    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            embedding_score = float(scores[i, j])
            if embedding_score < threshold:
                continue

            left = embedded[i]
            right = embedded[j]
            title_ratio = SequenceMatcher(
                None,
                left["normalized_title"],
                right["normalized_title"],
            ).ratio()
            ingredient_overlap = _jaccard(left["main_ingredients"], right["main_ingredients"])

            if title_ratio < 0.60 and ingredient_overlap < 0.35:
                continue

            reason = (
                "high_embedding_similarity; "
                f"title={title_ratio:.3f}; ingredients={ingredient_overlap:.3f}; "
                f"text=not_checked; embedding={embedding_score:.3f}"
            )
            edges.append((left["record_id"], right["record_id"], embedding_score, reason))

    return edges


def _connected_components(record_ids, edges):
    """Group duplicate pair edges into reviewable clusters."""
    parent = {record_id: record_id for record_id in record_ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, _, _ in edges:
        union(left, right)

    groups = {}
    for record_id in record_ids:
        groups.setdefault(find(record_id), []).append(record_id)

    return [members for members in groups.values() if len(members) > 1]


def _load_existing_review_values(out):
    """Preserve manual duplicate review columns when regenerating proposals."""
    path = Path(out)
    if not path.exists():
        return {}

    df = pd.read_csv(path).fillna("")
    if "source_image" not in df.columns:
        return {}

    values = {}
    for _, row in df.iterrows():
        dataset = str(row.get("dataset", ""))
        source = str(row.get("source_image", ""))
        key = f"{dataset}:{source}" if dataset else source
        values[key] = {
            "suggested_action": str(row.get("suggested_action", "")),
            "keep_candidate": str(row.get("keep_candidate", "")),
            "review_note": str(row.get("review_note", "")),
        }
    return values


def _parse_dataset_specs(specs):
    """Parse CLI dataset specs formatted as name:ocr_dir:metadata_dir[:review_csv]."""
    if not specs:
        return DEFAULT_DATASETS

    datasets = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(
                "Dataset specs must be name:ocr_dir:metadata_dir[:review_csv]"
            )

        dataset = {
            "name": parts[0],
            "ocr_dir": parts[1],
            "metadata_dir": parts[2],
        }
        if len(parts) == 4:
            dataset["review_csv"] = parts[3]
        datasets.append(dataset)

    return datasets


def build_duplicate_candidates(
    datasets=None,
    out="duplicate_candidates.csv",
    cache_path="recipe_embedding_cache.pkl",
    ocr_excerpt_chars=1200,
):
    """Generate an offline duplicate proposal CSV without modifying recipe data."""
    datasets = datasets or DEFAULT_DATASETS
    out = Path(out)
    df = load_recipe_dataframe(datasets)
    embeddings = _load_embedding_cache(cache_path)
    records = _build_records(df, embeddings, ocr_excerpt_chars)
    by_id = {record["record_id"]: record for record in records}

    edges = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            signal = _pair_duplicate_signal(left, right)
            if signal is None:
                continue
            score, reason = signal
            edges.append((left["record_id"], right["record_id"], score, reason))

    edges.extend(_embedding_edges(records))

    pair_reasons = {}
    for left_id, right_id, score, reason in edges:
        pair_reasons.setdefault(left_id, []).append((score, right_id, reason))
        pair_reasons.setdefault(right_id, []).append((score, left_id, reason))

    groups = _connected_components([record["record_id"] for record in records], edges)
    groups.sort(key=lambda members: min(by_id[record_id]["source_image"] for record_id in members))
    existing_values = _load_existing_review_values(out)
    rows = []

    for group_index, members in enumerate(groups, start=1):
        ranked_members = sorted(
            members,
            key=lambda record_id: (
                -len(by_id[record_id]["comparison_text"]),
                by_id[record_id]["dataset"],
                by_id[record_id]["source_image"],
            ),
        )
        keep_record_id = ranked_members[0]
        keep_source = by_id[keep_record_id]["source_image"]
        group_id = f"dup_{group_index:04d}"

        for record_id in ranked_members:
            record = by_id[record_id]
            best_matches = sorted(pair_reasons.get(record_id, []), reverse=True)[:3]
            best_score = best_matches[0][0] if best_matches else 0.0
            reason = " | ".join(
                f"matches {by_id[other_id]['dataset']}:{by_id[other_id]['source_image']} ({detail})"
                for _, other_id, detail in best_matches
            )
            previous_key = (
                f"{record['dataset']}:{record['source_image']}"
                if record["dataset"]
                else record["source_image"]
            )
            previous = existing_values.get(previous_key, {})

            if record_id == keep_record_id:
                default_action = "keep"
                default_keep = "yes"
            elif best_score >= 0.97:
                default_action = f"duplicate_of:{keep_source}"
                default_keep = "no"
            else:
                default_action = "needs_review"
                default_keep = ""

            rows.append({
                "duplicate_group_id": group_id,
                "source_image": record["source_image"],
                "dataset": record["dataset"],
                "title": record["title"],
                "title_confidence": record["title_confidence"],
                "main_ingredients": _join_list(record["main_ingredients"]),
                "short_description": record["short_description"],
                "semantic_summary": record["semantic_summary"],
                "duplicate_reason": reason,
                "similarity_score": round(best_score, 3),
                "suggested_action": previous.get("suggested_action") or default_action,
                "keep_candidate": previous.get("keep_candidate") or default_keep,
                "review_note": previous.get("review_note", ""),
            })

    candidates_df = pd.DataFrame(rows, columns=DUPLICATE_COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(out, index=False)

    print(
        f"Wrote {len(candidates_df)} duplicate candidate rows "
        f"across {len(groups)} groups to {out}"
    )
    return candidates_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate an offline duplicate proposal CSV for RecipeBox."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset spec: name:ocr_dir:metadata_dir[:review_csv]. Can be repeated.",
    )
    parser.add_argument("--out", default="duplicate_candidates.csv")
    parser.add_argument("--cache_path", default="recipe_embedding_cache.pkl")
    parser.add_argument("--ocr_excerpt_chars", type=int, default=1200)

    args = parser.parse_args()

    build_duplicate_candidates(
        datasets=_parse_dataset_specs(args.dataset),
        out=args.out,
        cache_path=args.cache_path,
        ocr_excerpt_chars=args.ocr_excerpt_chars,
    )


if __name__ == "__main__":
    main()
