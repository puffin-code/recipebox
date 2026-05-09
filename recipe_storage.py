import json
from pathlib import Path

import pandas as pd

RATINGS_CSV = "recipe_ratings.csv"


def _list_field(data, key):
    """Return list metadata fields safely for older JSON or null values."""
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def load_ocr_text_for_source(source_image):
    """Load OCR text for a recipe image name, returning blank text if missing."""
    ocr_path = Path("ocr_pages") / f"{Path(source_image).stem}.txt"

    if ocr_path.exists():
        return ocr_path.read_text(encoding="utf-8")

    return ""


def ocr_path_for_source(source_image):
    """Return the expected OCR text path for a recipe image name."""
    return Path("ocr_pages") / f"{Path(source_image).stem}.txt"


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


def save_rating(source_image, rating, personal_notes=""):
    """Upsert one recipe rating into the local ratings CSV."""
    path = Path(RATINGS_CSV)

    ratings_df = load_ratings()

    ratings_df = ratings_df[
        ratings_df["source_image"] != source_image
    ]

    ratings_df = pd.concat([
        ratings_df,
        pd.DataFrame([{
            "source_image": source_image,
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


def load_ratings():
    """Load ratings CSV and normalize older local schemas."""
    path = Path(RATINGS_CSV)

    if path.exists():
        ratings_df = pd.read_csv(path)
    else:
        ratings_df = pd.DataFrame(columns=[
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

    return ratings_df[["source_image", "rating", "personal_notes"]]


def load_metadata(metadata_dir="recipe_metadata"):
    """Load generated metadata JSON files into a recipe dataframe."""
    rows = []

    for path in sorted(Path(metadata_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        rows.append({
            "source_image": data.get("source_image") or path.stem,
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
            "image_path": f"Photos-67/{data.get('source_image')}",
        })

    return pd.DataFrame(rows)


def load_review_notes(review_csv="recipe_review_inventory.csv"):
    """Load manual review notes when present."""
    path = Path(review_csv)

    if not path.exists():
        return pd.DataFrame(columns=["source_image", "review_note"])

    review_df = pd.read_csv(path)

    if "review_note" not in review_df.columns:
        review_df["review_note"] = ""

    return review_df[["source_image", "review_note"]]


def load_recipe_dataframe():
    """Load metadata, ratings, and browse flags for the Streamlit app."""
    df = load_metadata()

    load_review_notes()

    ratings_df = load_ratings()

    df = df.merge(
        ratings_df,
        on="source_image",
        how="left",
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


def filter_recipe_dataframe(df, search, favorites_only):
    """Apply the existing title/type/description/ingredient and favorite filters."""
    if search:
        s = search.lower()

        mask = (
            df["title"].str.lower().str.contains(s, na=False)
            |
            df["dish_type"].str.lower().str.contains(s, na=False)
            |
            df["short_description"].str.lower().str.contains(s, na=False)
            |
            df["main_ingredients"].apply(lambda xs: s in " ".join(xs).lower())
        )

        filtered = df[mask]
    else:
        filtered = df

    if favorites_only:
        filtered = filtered[filtered["favorite"]]

    return filtered
