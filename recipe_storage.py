import json
import re
from pathlib import Path

import pandas as pd

from recipe_config import data_path, dataset_path

RATINGS_CSV = data_path("recipe_ratings.csv")
DEFAULT_DATASETS = [
    {
        "name": "original",
        "ocr_dir": dataset_path("ocr_pages"),
        "metadata_dir": dataset_path("recipe_metadata"),
        "review_csv": dataset_path("recipe_review_inventory.csv"),
    }
]

SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "of",
    "or",
    "recipe",
    "recipes",
    "the",
    "to",
    "with",
}
NEGATIVE_PREFIXES = ("no", "without", "avoid", "exclude")


def _list_field(data, key):
    """Return list metadata fields safely for older JSON or null values."""
    value = data.get(key, [])
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return value if isinstance(value, list) else []


def record_id_for(dataset, source_image):
    """Build a stable row id that is safe across datasets."""
    return f"{dataset}:{source_image}"


def load_ocr_text_for_source(source_image, ocr_dir=dataset_path("ocr_pages")):
    """Load OCR text for a recipe image name, returning blank text if missing."""
    ocr_path = Path(ocr_dir) / f"{Path(source_image).stem}.txt"

    if ocr_path.exists():
        return ocr_path.read_text(encoding="utf-8")

    return ""


def ocr_path_for_source(source_image, ocr_dir=dataset_path("ocr_pages")):
    """Return the expected OCR text path for a recipe image name."""
    return Path(ocr_dir) / f"{Path(source_image).stem}.txt"


def detect_favorite(notes):
    """Infer favorites from handwritten/user note language."""
    if isinstance(notes, list):
        text = " ".join(notes).lower()
    else:
        text = str(notes).lower()

    favorite_terms = [
        "5/5",
        "favorite",
        "favourite",
        "really good",
        "excellent",
        "make again",
    ]

    return any(term in text for term in favorite_terms)


def save_rating(record_id, rating, personal_notes="", source_image=None):
    """Upsert one recipe rating into the local ratings CSV."""
    path = Path(RATINGS_CSV)

    ratings_df = load_ratings()

    ratings_df = ratings_df[
        ratings_df["record_id"] != record_id
    ]

    ratings_df = pd.concat([
        ratings_df,
        pd.DataFrame([{
            "record_id": record_id,
            "source_image": source_image or record_id,
            "rating": rating,
            "personal_notes": personal_notes,
        }]),
    ], ignore_index=True)

    ratings_df.to_csv(path, index=False)


def format_user_notes(notes):
    """Render metadata notes as compact display text."""
    if isinstance(notes, list):
        return "; ".join(notes)
    if pd.isna(notes):
        return ""
    return str(notes)


def _format_list_field(value):
    """Render list or string metadata fields for search."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if pd.isna(value):
        return ""
    return str(value)


def _search_tokens(text):
    """Normalize searchable words with light singular/plural handling."""
    tokens = []
    for token in re.findall(r"[a-z0-9]+", str(text).lower()):
        if len(token) < 2 or token in SEARCH_STOPWORDS:
            continue
        tokens.append(token)
        if token.endswith("ies") and len(token) > 4:
            tokens.append(f"{token[:-3]}y")
        elif token.endswith("es") and len(token) > 3:
            tokens.append(token[:-2])
        elif token.endswith("s") and len(token) > 3:
            tokens.append(token[:-1])
    return tokens


def parse_search_query(search):
    """Split a query into positive text plus explicit exclusion terms."""
    query = str(search or "").strip()
    if not query:
        return "", []

    prefix_pattern = "|".join(NEGATIVE_PREFIXES)
    clause_pattern = re.compile(
        rf"\b(?:{prefix_pattern})\s+(.+?)"
        rf"(?=(?:\s*[,;]\s*|\s+\b(?:{prefix_pattern})\b|$))",
        flags=re.IGNORECASE,
    )

    excluded_terms = []
    for match in clause_pattern.finditer(query):
        clause = re.sub(r"\s+", " ", match.group(1)).strip(" ,;")
        clause = re.sub(r"\b(?:and|or)$", "", clause, flags=re.IGNORECASE).strip()
        for term in re.split(r"\s+(?:and|or)\s+|[,;]", clause, flags=re.IGNORECASE):
            normalized = term.strip()
            if normalized:
                excluded_terms.append(normalized)

    positive_query = clause_pattern.sub(" ", query)
    positive_query = re.sub(r"\s+", " ", positive_query).strip(" ,;")
    return positive_query, excluded_terms


def text_contains_search_term(text, term):
    """Match whole terms with light plural handling, avoiding partial words."""
    text_tokens = set(_search_tokens(text))
    term_words = re.findall(r"[a-z0-9]+", str(term).lower())
    term_groups = [
        set(_search_tokens(word))
        for word in term_words
        if set(_search_tokens(word))
    ]
    return bool(text_tokens and term_groups) and all(
        variants & text_tokens
        for variants in term_groups
    )


def row_search_text(row):
    """Build the metadata-only text used by browse filtering."""
    return " ".join([
        str(row.get("title", "")),
        str(row.get("source_image", "")),
        str(row.get("record_id", "")),
        str(row.get("dataset", "")),
        str(row.get("dish_type", "")),
        str(row.get("short_description", "")),
        str(row.get("semantic_summary", "")),
        _format_list_field(row.get("main_ingredients", [])),
        format_user_notes(row.get("user_notes", [])),
    ])


def _ingredient_match(ingredients, search):
    """Match ingredient searches against list/string metadata robustly."""
    ingredient_text = _format_list_field(ingredients).lower()
    if not ingredient_text:
        return False

    normalized_search = str(search).lower().strip()
    if normalized_search and normalized_search in ingredient_text:
        return True

    ingredient_tokens = set(_search_tokens(ingredient_text))
    return any(token in ingredient_tokens for token in _search_tokens(search))


def _normalize_ingredient_phrase(value):
    """Normalize one ingredient phrase for exact matching."""
    return " ".join(_search_tokens(value))


def _exact_ingredient_match(ingredients, search):
    """Match only a whole ingredient phrase, e.g. corn but not corn flour."""
    search_phrase = _normalize_ingredient_phrase(search)
    if not search_phrase:
        return False

    if isinstance(ingredients, list):
        ingredient_phrases = ingredients
    elif pd.isna(ingredients):
        ingredient_phrases = []
    else:
        ingredient_phrases = re.split(r"[,;/]", str(ingredients))

    return any(
        _normalize_ingredient_phrase(ingredient) == search_phrase
        for ingredient in ingredient_phrases
    )


def load_ratings():
    """Load ratings CSV and normalize older local schemas."""
    path = Path(RATINGS_CSV)

    if path.exists():
        ratings_df = pd.read_csv(path)
    else:
        ratings_df = pd.DataFrame(columns=[
            "record_id",
            "source_image",
            "rating",
            "personal_notes",
        ])

    # Repair older CSVs that used "notes".
    if "notes" in ratings_df.columns and "personal_notes" not in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={"notes": "personal_notes"})

    if "rating" not in ratings_df.columns:
        ratings_df["rating"] = pd.NA

    if "personal_notes" not in ratings_df.columns:
        ratings_df["personal_notes"] = ""

    if "source_image" not in ratings_df.columns:
        ratings_df["source_image"] = ratings_df.get("record_id", "")

    if "record_id" not in ratings_df.columns:
        ratings_df["record_id"] = ratings_df["source_image"]

    return ratings_df[["record_id", "source_image", "rating", "personal_notes"]]


def load_metadata(
    metadata_dir=dataset_path("recipe_metadata"),
    dataset="original",
    ocr_dir=dataset_path("ocr_pages"),
):
    """Load generated metadata JSON files into a recipe dataframe."""
    rows = []

    for path in sorted(Path(metadata_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        source_image = data.get("source_image") or path.stem

        rows.append({
            "dataset": dataset,
            "ocr_dir": str(ocr_dir),
            "metadata_dir": str(metadata_dir),
            "record_id": record_id_for(dataset, source_image),
            "source_image": source_image,
            "title": data.get("title") or "Untitled recipe/page",
            "title_confidence": data.get("title_confidence"),
            "recipe_structure": data.get("recipe_structure", "unknown"),
            "dish_type": data.get("dish_type") or "unknown",
            "main_ingredients": _list_field(data, "main_ingredients"),
            "short_description": data.get("short_description") or "",
            "semantic_summary": data.get("semantic_summary") or "",
            "vibe_tags": _list_field(data, "vibe_tags"),
            "season_tags": _list_field(data, "season_tags"),
            "meal_context_tags": _list_field(data, "meal_context_tags"),
            "effort_level": data.get("effort_level") or "unknown",
            "served_temperature": data.get("served_temperature") or "unknown",
            "make_ahead_potential": data.get("make_ahead_potential") or "unknown",
            "user_notes": _list_field(data, "user_notes"),
            "image_path": "",
        })

    return pd.DataFrame(rows)


def load_review_notes(
    review_csv=dataset_path("recipe_review_inventory.csv"),
    dataset="original",
):
    """Load manual review notes when present."""
    path = Path(review_csv)

    if not path.exists():
        return pd.DataFrame(columns=["record_id", "source_image", "review_note"])

    review_df = pd.read_csv(path)

    if "review_note" not in review_df.columns:
        review_df["review_note"] = ""

    if "source_image" not in review_df.columns:
        review_df["source_image"] = ""

    if "record_id" not in review_df.columns:
        review_df["record_id"] = review_df["source_image"].apply(
            lambda source: record_id_for(dataset, source)
        )

    return review_df[["record_id", "source_image", "review_note"]]


def load_recipe_dataframe(datasets=None):
    """Load metadata, ratings, and browse flags for the Streamlit app."""
    datasets = datasets or DEFAULT_DATASETS

    metadata_frames = []
    review_frames = []

    for dataset in datasets:
        name = dataset["name"]
        metadata_frames.append(load_metadata(
            metadata_dir=dataset["metadata_dir"],
            dataset=name,
            ocr_dir=dataset["ocr_dir"],
        ))
        if dataset.get("review_csv"):
            review_frames.append(load_review_notes(
                review_csv=dataset["review_csv"],
                dataset=name,
            ))

    df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    review_df = (
        pd.concat(review_frames, ignore_index=True)
        if review_frames
        else pd.DataFrame(columns=["record_id", "review_note"])
    )

    ratings_df = load_ratings()

    df = df.merge(
        review_df[["record_id", "review_note"]],
        on="record_id",
        how="left",
    )

    df = df.merge(
        ratings_df[["record_id", "rating", "personal_notes"]],
        on="record_id",
        how="left",
    )

    legacy_ratings = ratings_df[ratings_df["record_id"] == ratings_df["source_image"]]
    if not legacy_ratings.empty:
        legacy_ratings = legacy_ratings.drop_duplicates("source_image").set_index("source_image")
        missing_rating = df["rating"].isna()
        df.loc[missing_rating, "rating"] = df.loc[missing_rating, "source_image"].map(
            legacy_ratings["rating"]
        )
        missing_notes = df["personal_notes"].isna()
        df.loc[missing_notes, "personal_notes"] = df.loc[missing_notes, "source_image"].map(
            legacy_ratings["personal_notes"]
        )

    if "review_note" not in df.columns:
        df["review_note"] = ""

    df["review_note"] = df["review_note"].fillna("")
    df["personal_notes"] = df["personal_notes"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df["favorite"] = (
        df["user_notes"].apply(detect_favorite)
        |
        (df["rating"] >= 4)
    )

    df["hide_from_browse"] = (
        df["review_note"].str.contains("combine_with", case=False, na=False)
        |
        df["recipe_structure"].isin(["continuation_page", "partial_recipe"])
    )

    return df


def filter_recipe_dataframe(df, search, favorites_only, search_mode="all"):
    """Apply browser text and favorite filters without touching the LLM path."""
    positive_search, excluded_terms = parse_search_query(search)

    if positive_search:
        s = positive_search.lower()
        ingredient_matcher = (
            _exact_ingredient_match
            if search_mode == "ingredients_exact"
            else _ingredient_match
        )
        ingredient_mask = df["main_ingredients"].apply(
            lambda ingredients: ingredient_matcher(ingredients, positive_search)
        )

        if search_mode in {"ingredients_broad", "ingredients_exact"}:
            mask = ingredient_mask
        else:
            mask = (
                df["title"].str.lower().str.contains(s, na=False)
                |
                df["source_image"].astype(str).str.lower().str.contains(s, na=False)
                |
                df["record_id"].astype(str).str.lower().str.contains(s, na=False)
                |
                df["dataset"].astype(str).str.lower().str.contains(s, na=False)
                |
                df["dish_type"].str.lower().str.contains(s, na=False)
                |
                df["short_description"].str.lower().str.contains(s, na=False)
                |
                df["semantic_summary"].str.lower().str.contains(s, na=False)
                |
                ingredient_mask
                |
                df["user_notes"].apply(lambda notes: s in format_user_notes(notes).lower())
            )

        filtered = df[mask]
    else:
        filtered = df

    if excluded_terms:
        # Negative clauses are hard filters, not positive search words.
        excluded_mask = filtered.apply(
            lambda row: any(
                text_contains_search_term(row_search_text(row), term)
                for term in excluded_terms
            ),
            axis=1,
        )
        filtered = filtered[~excluded_mask]

    if favorites_only:
        filtered = filtered[filtered["favorite"]]

    return filtered
