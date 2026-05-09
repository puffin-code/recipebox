import json
from pathlib import Path

import pandas as pd
import streamlit as st

from openai import OpenAI
from dotenv import load_dotenv

from test_setup import ocr_image, extract_recipe_metadata, cosine_similarity, expand_query

load_dotenv()
client = OpenAI()

st.set_page_config(
    page_title="Recipe Browser",
    page_icon="🍳",
    layout="wide",
)

st.sidebar.header("Filters")

favorites_only = st.sidebar.checkbox(
    "⭐ Favorites only",
    value=False
)

st.sidebar.header("Add recipe")

uploaded_image = st.sidebar.file_uploader(
    "Upload recipe photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    if st.sidebar.button("OCR and add recipe"):
        image_bytes = uploaded_image.getvalue()
        image_name = uploaded_image.name

        # save uploaded image
        upload_dir = Path("uploaded_recipe_images")
        upload_dir.mkdir(exist_ok=True)

        image_path = upload_dir / image_name
        image_path.write_bytes(image_bytes)

        st.sidebar.info("Running OCR...")

        recipe_text = ocr_image(image_path)

        # save OCR text
        ocr_dir = Path("ocr_pages")
        ocr_dir.mkdir(exist_ok=True)

        txt_path = ocr_dir / f"{image_path.stem}.txt"
        txt_path.write_text(recipe_text, encoding="utf-8")

        # save metadata
        metadata = extract_recipe_metadata(
            recipe_text,
            source_image=image_name
        )

        metadata_dir = Path("recipe_metadata")
        metadata_dir.mkdir(exist_ok=True)

        json_path = metadata_dir / f"{image_path.stem}.json"
        json_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        st.sidebar.success(f"Added: {metadata.get('title', image_name)}")
        st.rerun()

RATINGS_CSV = "recipe_ratings.csv"


def recommend_from_cookbook(user_query, df, max_candidates=12, top_n=6):
    candidates = retrieve_recipe_candidates(
        user_query,
        df,
        top_k=max_candidates
    )

    candidate_blocks = []

    for item in candidates:
        row = item["row"]
        source = row["source_image"]
        ocr_text = load_ocr_text_for_source(source)

        candidate_blocks.append(
            f"""
Page: {source}
Title: {row['title']}
Dish type: {row['dish_type']}
Main ingredients: {", ".join(row["main_ingredients"])}
Description: {row["short_description"]}
User notes: {format_user_notes(row["user_notes"])}

Recipe text excerpt:
{ocr_text[:1500]}
"""
        )

    context = "\n\n---\n\n".join(candidate_blocks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
               "content": (
                    "You are a helpful cookbook assistant. "
                    "Recommend recipes ONLY from the provided cookbook candidates. "
                    "Do not invent recipes. "
                    f"Recommend the top {top_n} recipes if enough good matches are available. "
                    "For each recommendation, include the recipe title/page and a short reason. "
                    "Prefer recipes that best match the user's mood, ingredients, season, or cooking intent."
                )
            },
            {
                "role": "user",
                "content": f"Cookbook candidates:\n{context}\n\nRequest:\n{user_query}",
            },
        ],
    )

    return response.choices[0].message.content



def build_recipe_search_text(row):
    ocr_text = load_ocr_text_for_source(row["source_image"])

    parts = [
        f"Title: {row.get('title', '')}",
        f"Dish type: {row.get('dish_type', '')}",
        f"Main ingredients: {', '.join(row.get('main_ingredients', []))}",
        f"Description: {row.get('short_description', '')}",
        f"User notes: {format_user_notes(row.get('user_notes', ''))}",
        f"Recipe text: {ocr_text[:2500]}",
    ]

    return "\n".join(parts)


def retrieve_recipe_candidates(user_query, df, top_k=8):
    expanded_queries = [user_query] + expand_query(user_query)

    recipe_texts = []

    for _, row in df.iterrows():
        recipe_texts.append({
            "source_image": row["source_image"],
            "title": row["title"],
            "search_text": build_recipe_search_text(row),
            "row": row,
        })

    # embed recipes
    recipe_embeddings = []

    for item in recipe_texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=item["search_text"]
        ).data[0].embedding

        recipe_embeddings.append({
            **item,
            "embedding": emb,
        })

    all_scores = []

    for q in expanded_queries:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=q
        ).data[0].embedding

        for item in recipe_embeddings:
            score = cosine_similarity(q_emb, item["embedding"])
            all_scores.append((score, item))

    # keep best score per page
    best = {}

    for score, item in all_scores:
        source = item["source_image"]

        if source not in best or score > best[source][0]:
            best[source] = (score, item)

    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)

    return [item for score, item in ranked[:top_k]]


def load_ocr_text_for_source(source_image):
    ocr_path = Path("ocr_pages") / f"{Path(source_image).stem}.txt"

    if ocr_path.exists():
        return ocr_path.read_text(encoding="utf-8")

    return ""

def detect_favorite(notes):

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
            "personal_notes": personal_notes
        }])
    ], ignore_index=True)

    ratings_df.to_csv(path, index=False)

def format_user_notes(notes):
    if isinstance(notes, list):
        return "; ".join(notes)
    if pd.isna(notes):
        return ""
    return str(notes)



def load_ratings():
    path = Path(RATINGS_CSV)

    if path.exists():
        ratings_df = pd.read_csv(path)
    else:
        ratings_df = pd.DataFrame(columns=[
            "source_image",
            "rating",
            "personal_notes"
        ])

    # repair older CSVs that used "notes"
    if "notes" in ratings_df.columns and "personal_notes" not in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={"notes": "personal_notes"})

    if "rating" not in ratings_df.columns:
        ratings_df["rating"] = pd.NA

    if "personal_notes" not in ratings_df.columns:
        ratings_df["personal_notes"] = ""

    return ratings_df[["source_image", "rating", "personal_notes"]]



def load_metadata(metadata_dir="recipe_metadata"):
    rows = []

    for path in sorted(Path(metadata_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
       

        rows.append({
            "source_image": data.get("source_image") or path.stem,
            "title": data.get("title") or "Untitled recipe/page",
            "title_confidence": data.get("title_confidence"),
            "recipe_structure": data.get("recipe_structure", "unknown"),
            "dish_type": data.get("dish_type") or "unknown",
            "main_ingredients": data.get("main_ingredients", []),
            "short_description": data.get("short_description") or "",
            "user_notes": data.get("user_notes", []),
            "image_path": f"Photos-67/{data.get('source_image')}",
        })

        

    return pd.DataFrame(rows)



def load_review_notes(review_csv="recipe_review_inventory.csv"):
    path = Path(review_csv)

    if not path.exists():
        return pd.DataFrame(columns=["source_image", "review_note"])

    review_df = pd.read_csv(path)

    if "review_note" not in review_df.columns:
        review_df["review_note"] = ""

    return review_df[["source_image", "review_note"]]

df = load_metadata()

review_df = load_review_notes()

ratings_df = load_ratings()

df = df.merge(
    ratings_df,
    on="source_image",
    how="left"
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



show_hidden = st.checkbox("Show continuation/partial pages", value=False)

if not show_hidden:
    df = df[~df["hide_from_browse"]]

st.title("🍳 Recipe Browser")

st.write(f"Loaded **{len(df)}** recipe pages.")

st.subheader("🔎 Ask the cookbook")

global_query = st.text_input(
    "Ask for recommendations",
    placeholder=(
        "Try: warming recipes for a cold day, "
        "refreshing lemony options for summer, "
        "something easy but impressive"
    ),
    key="global_recommendation_query",
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
)

if global_query:
    with st.spinner("Searching the cookbook..."):
        recommendation = recommend_from_cookbook(
            global_query,
            df,
            max_candidates=max(top_n * 2, 8),
            top_n=top_n,
        )

    st.markdown(recommendation)

st.divider()



search = st.text_input("Search recipes", placeholder="Try: aubergine, chickpeas, salad, dessert")

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


st.write(f"Showing **{len(filtered)}** results.")
filtered = filtered.sort_values("title")
for _, row in filtered.iterrows():
    with st.container(border=True):
        st.subheader(row["title"])

        if row["short_description"]:
            st.write(row["short_description"])

        st.caption(
            f"Page: {row['source_image']} | "
            f"Type: {row['dish_type']} | "
            f"Structure: {row['recipe_structure']}"
        )

        if row["main_ingredients"]:
            st.write("**Main ingredients:** " + ", ".join(row["main_ingredients"]))

        notes = format_user_notes(row["user_notes"])
        if notes:
            st.success(f"⭐ Notes: {notes}")

        ocr_path = Path("ocr_pages") / f"{Path(row['source_image']).stem}.txt"

        with st.expander("View recipe text"):
            if ocr_path.exists():
                st.text(ocr_path.read_text(encoding="utf-8"))
            else:
                st.warning("Recipe text not found.")

        with st.expander("Ask about this recipe"):
            question = st.text_input(
                "Question",
                key=f"question_{row['source_image']}",
                placeholder="Try: How can I prep this ahead? What can I substitute?"
            )

            if question:
                if ocr_path.exists():
                    recipe_text = ocr_path.read_text(encoding="utf-8")

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful cooking assistant. "
                                    "Use the provided recipe text as the main context. "
                                    "You may use general cooking knowledge for substitutions, prep advice, "
                                    "and practical suggestions. Keep answers concise and useful."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Recipe:\n{recipe_text}\n\nQuestion:\n{question}",
                            },
                        ],
                    )

                    st.write(response.choices[0].message.content)
                else:
                    st.warning("Recipe text not found.")

        with st.expander("Rate / personal notes"):

            rating = st.radio(
                "Rating",
                options=[1, 2, 3, 4, 5],
                horizontal=True,
                key=f"rating_{row['source_image']}"
            )

            personal_note = st.text_input(
                "Personal note",
                value="",
                key=f"note_{row['source_image']}"
            )

            if st.button(
                "Save rating",
                key=f"save_{row['source_image']}"
            ):
                save_rating(
                    row["source_image"],
                    rating,
                    personal_note
                )

                st.success("Saved!")