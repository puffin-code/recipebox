import streamlit as st

from recipe_ingest import ingest_uploaded_recipe
from recipe_search import (
    answer_recipe_question,
    recommend_from_cookbook,
    refresh_recipe_embedding_cache,
    retrieve_ranked_recipes,
)
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


def dataframe_cache_key(df):
    """Summarize recipe metadata for Streamlit retrieval caching."""
    def as_tuple(value):
        return tuple(value) if isinstance(value, list) else ()

    return tuple(
        (
            row["source_image"],
            row["title"],
            tuple(row["main_ingredients"]),
            row["short_description"],
            row.get("semantic_summary", ""),
            as_tuple(row.get("vibe_tags", [])),
            as_tuple(row.get("season_tags", [])),
            as_tuple(row.get("meal_context_tags", [])),
            row.get("effort_level", ""),
            row.get("served_temperature", ""),
            row.get("make_ahead_potential", ""),
            as_tuple(row.get("user_notes", [])),
        )
        for _, row in df.iterrows()
    )


@st.cache_data(show_spinner=False)
def cached_ranked_recipes(query, top_k, df_key, _df):
    """Cache query results across Streamlit reruns."""
    return retrieve_ranked_recipes(query, _df, top_k=top_k)


if "browser_search" not in st.session_state:
    # Browser filtering is independent from global recommendation form state.
    st.session_state["browser_search"] = st.session_state.get("recipe_browser_search", "")

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
        st.session_state["refresh_search_index_after_upload"] = True
        st.rerun()

df = load_recipe_dataframe()

show_hidden = st.checkbox("Show continuation/partial pages", value=False)

if not show_hidden:
    df = df[~df["hide_from_browse"]]

st.sidebar.header("Search index")

if st.session_state.pop("refresh_search_index_after_upload", False):
    with st.sidebar.status("Refreshing search index...", expanded=False):
        stats = refresh_recipe_embedding_cache(df, force=False)
        cached_ranked_recipes.clear()
    st.sidebar.success(
        "Search index refreshed: "
        f"{stats['reused']} reused, {stats['generated']} generated."
    )

if st.sidebar.button("Refresh search index"):
    with st.sidebar.status("Refreshing search index...", expanded=False):
        stats = refresh_recipe_embedding_cache(df, force=False)
        cached_ranked_recipes.clear()
    st.sidebar.success(
        "Search index refreshed: "
        f"{stats['reused']} reused, "
        f"{stats['generated']} generated, "
        f"{stats['removed_stale']} removed."
    )

with st.sidebar.expander("Rebuild search index from scratch"):
    st.warning("This regenerates every recipe embedding and may cost more.")
    confirm_rebuild = st.checkbox("I understand", key="confirm_rebuild_search_index")
    if st.button("Rebuild search index", disabled=not confirm_rebuild):
        with st.status("Rebuilding search index...", expanded=False):
            stats = refresh_recipe_embedding_cache(df, force=True)
            cached_ranked_recipes.clear()
        st.success(
            "Search index rebuilt: "
            f"{stats['generated']} generated, "
            f"{stats['removed_stale']} removed."
        )

st.title("🍳 Recipe Browser")

st.write(f"Loaded **{len(df)}** recipe pages.")

st.subheader("🔎 Ask the cookbook")

with st.form("global_recommendation_form"):
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
        max_value=15,
        value=6,
        step=1,
        key="global_recommendation_top_n",
    )

    submitted_recommendation = st.form_submit_button("Search cookbook")

# Streamlit reruns the whole script often; only this submit branch may call the
# global retrieval/LLM path. Later reruns simply redraw stored results.
if submitted_recommendation:
    cleaned_query = global_query.strip()

    if cleaned_query:
        with st.spinner("Searching the cookbook..."):
            ranked_matches = cached_ranked_recipes(
                cleaned_query,
                max(30, top_n * 3),
                dataframe_cache_key(df),
                df,
            )
            if ranked_matches.empty:
                recommendation = ""
            else:
                recommendation = recommend_from_cookbook(
                    cleaned_query,
                    ranked_matches,
                    top_n=top_n,
                )

        st.session_state["last_recommendation"] = {
            "query": cleaned_query,
            "top_n": top_n,
            "markdown": recommendation,
            "ranked_matches": ranked_matches,
        }
    else:
        st.warning("Enter a cookbook question before searching.")

last_recommendation = st.session_state.get("last_recommendation")

if last_recommendation:
    recommendation = last_recommendation.get("markdown", "")
    ranked_matches = last_recommendation.get("ranked_matches")
    last_query = last_recommendation.get("query", "")

    if last_query:
        st.caption(
            f"Last recommendation search: {last_query} "
            f"({last_recommendation.get('top_n')} requested)"
        )

    if recommendation:
        st.markdown(recommendation)
    else:
        st.warning("No matching recipes found.")

    if ranked_matches is not None and not ranked_matches.empty:
        st.subheader("All ranked matches")
        with st.container(height=420, border=True):
            for _, row in ranked_matches.iterrows():
                description = row["short_description"] or row.get("semantic_summary", "")

                st.markdown(
                    f"**{int(row['rank'])}. {row['title']}**  "
                    f"`{row['score']:.3f}`"
                )
                st.caption(f"Page: {row['source_image']}")

                if description:
                    st.write(description)

                if row["main_ingredients"]:
                    st.write("**Main ingredients:** " + ", ".join(row["main_ingredients"]))

                if st.button("View recipe", key=f"view_ranked_{row['source_image']}"):
                    st.session_state["browser_search"] = row["source_image"]
                    st.rerun()

                st.divider()

st.divider()

search = st.text_input(
    "Search recipes",
    placeholder="Try: aubergine, chickpeas, salad, dessert",
    key="browser_search",
)
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
