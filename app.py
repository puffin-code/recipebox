import streamlit as st

from recipe_ingest import ingest_uploaded_recipe
from recipe_search import answer_recipe_question, recommend_from_cookbook
from recipe_storage import (
    filter_recipe_dataframe,
    format_user_notes,
    load_recipe_dataframe,
    ocr_path_for_source,
    save_rating,
)

st.set_page_config(
    page_title="Recipe Browser",
    page_icon="🍳",
    layout="wide",
)

st.sidebar.header("Filters")

favorites_only = st.sidebar.checkbox(
    "⭐ Favorites only",
    value=False,
)

st.sidebar.header("Add recipe")

uploaded_image = st.sidebar.file_uploader(
    "Upload recipe photo",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    if st.sidebar.button("OCR and add recipe"):
        image_bytes = uploaded_image.getvalue()
        image_name = uploaded_image.name

        st.sidebar.info("Running OCR...")

        metadata = ingest_uploaded_recipe(image_bytes, image_name)

        st.sidebar.success(f"Added: {metadata.get('title', image_name)}")
        st.rerun()

df = load_recipe_dataframe()

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
filtered = filter_recipe_dataframe(df, search, favorites_only)

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

        ocr_path = ocr_path_for_source(row["source_image"])

        with st.expander("View recipe text"):
            if ocr_path.exists():
                st.text(ocr_path.read_text(encoding="utf-8"))
            else:
                st.warning("Recipe text not found.")

        with st.expander("Ask about this recipe"):
            question = st.text_input(
                "Question",
                key=f"question_{row['source_image']}",
                placeholder="Try: How can I prep this ahead? What can I substitute?",
            )

            if question:
                if ocr_path.exists():
                    recipe_text = ocr_path.read_text(encoding="utf-8")
                    st.write(answer_recipe_question(recipe_text, question))
                else:
                    st.warning("Recipe text not found.")

        with st.expander("Rate / personal notes"):

            rating = st.radio(
                "Rating",
                options=[1, 2, 3, 4, 5],
                horizontal=True,
                key=f"rating_{row['source_image']}",
            )

            personal_note = st.text_input(
                "Personal note",
                value="",
                key=f"note_{row['source_image']}",
            )

            if st.button(
                "Save rating",
                key=f"save_{row['source_image']}",
            ):
                save_rating(
                    row["source_image"],
                    rating,
                    personal_note,
                )

                st.success("Saved!")
