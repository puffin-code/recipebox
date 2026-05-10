from html import escape

import pandas as pd
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
    page_title="RecipeBox",
    page_icon="🍳",
    layout="wide",
)


DATASETS = [
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


def add_page_styles():
    """Add lightweight visual polish without changing Streamlit behavior."""
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid #eadfce;
            border-radius: 8px;
            background: #fffaf3;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.15;
        }
        .hero p {
            margin: .35rem 0 0;
            color: #65584a;
            font-size: 1rem;
        }
        .chip {
            display: inline-block;
            padding: .18rem .52rem;
            margin: .12rem .2rem .12rem 0;
            border: 1px solid #e6d8c4;
            border-radius: 999px;
            background: #fffaf3;
            color: #5e5143;
            font-size: .82rem;
            line-height: 1.4;
        }
        .recipe-meta {
            color: #6e6257;
            font-size: .88rem;
            margin-bottom: .35rem;
        }
        .soft-note {
            padding: .6rem .75rem;
            border-left: 3px solid #e0b36a;
            background: #fffaf3;
            color: #55483b;
            border-radius: 4px;
            margin: .5rem 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #eadfce;
            box-shadow: 0 1px 2px rgba(74, 55, 35, .05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dataframe_cache_key(df):
    """Summarize recipe metadata for Streamlit retrieval caching."""
    def as_tuple(value):
        return tuple(value) if isinstance(value, list) else ()

    return tuple(
        (
            row["source_image"],
            row.get("record_id", ""),
            row.get("dataset", ""),
            row.get("ocr_dir", ""),
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


def chip_html(label):
    """Render one compact metadata chip."""
    if label is None or label == "" or label == "unknown":
        return ""
    return f'<span class="chip">{escape(str(label))}</span>'


def render_chips(labels):
    """Render a row of chips from a list of labels."""
    chips = "".join(chip_html(label) for label in labels if label)
    if chips:
        st.markdown(chips, unsafe_allow_html=True)


def metadata_tags(row):
    """Collect existing metadata fields that are useful as visual tags."""
    tags = [
        row.get("dish_type"),
        row.get("effort_level"),
        row.get("served_temperature"),
        f"make ahead: {row.get('make_ahead_potential')}"
        if row.get("make_ahead_potential") not in ("", "unknown", None)
        else "",
    ]
    tags.extend(row.get("season_tags", [])[:3])
    tags.extend(row.get("meal_context_tags", [])[:3])
    return tags


def recipe_summary(row):
    """Prefer the short human description, then semantic summary."""
    return row.get("short_description") or row.get("semantic_summary") or ""


def rating_label(row):
    """Format saved rating state without changing rating persistence."""
    rating = row.get("rating")
    if pd.notna(rating):
        return f"Rated {int(rating)}/5"
    if row.get("favorite"):
        return "Favorite"
    return ""


def render_recipe_card(row, key_prefix="recipe"):
    """Render one browsable recipe card with existing expanders/actions."""
    with st.container(border=True):
        title_col, tag_col = st.columns([0.75, 0.25])
        with title_col:
            st.subheader(row["title"])
        with tag_col:
            label = rating_label(row)
            if label:
                st.markdown(f"**{label}**")

        st.markdown(
            "<div class='recipe-meta'>"
            f"Dataset: {row['dataset']} &nbsp;|&nbsp; "
            f"Page: {row['source_image']} &nbsp;|&nbsp; "
            f"Structure: {row['recipe_structure']}"
            "</div>",
            unsafe_allow_html=True,
        )

        summary = recipe_summary(row)
        if summary:
            st.write(summary)

        render_chips(metadata_tags(row))

        if row["main_ingredients"]:
            st.markdown("**Main ingredients**")
            render_chips(row["main_ingredients"][:12])

        notes = format_user_notes(row["user_notes"])
        if notes:
            st.markdown(f"<div class='soft-note'>Saved note: {notes}</div>", unsafe_allow_html=True)

        ocr_path = ocr_path_for_source(row["source_image"], row.get("ocr_dir", "ocr_pages"))

        with st.expander("View recipe text"):
            if ocr_path.exists():
                st.text(ocr_path.read_text(encoding="utf-8"))
            else:
                st.warning("Recipe text not found.")

        with st.expander("Ask about this recipe"):
            question = st.text_input(
                "Question",
                key=f"question_{key_prefix}_{row['record_id']}",
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
                key=f"rating_{key_prefix}_{row['record_id']}",
            )

            personal_note = st.text_input(
                "Personal note",
                value="",
                key=f"note_{key_prefix}_{row['record_id']}",
            )

            if st.button(
                "Save rating",
                key=f"save_{key_prefix}_{row['record_id']}",
            ):
                save_rating(
                    row["record_id"],
                    rating,
                    personal_note,
                    source_image=row["source_image"],
                )
                st.success("Saved!")


def render_ranked_matches(ranked_matches):
    """Render compact ranked recommendation cards."""
    st.subheader("All ranked matches")
    with st.container(height=430, border=True):
        for _, row in ranked_matches.iterrows():
            with st.container(border=True):
                left, right = st.columns([0.78, 0.22])
                with left:
                    st.markdown(f"**{int(row['rank'])}. {row['title']}**")
                    st.caption(
                        f"Dataset: {row['dataset']} | "
                        f"Page: {row['source_image']} | "
                        f"Score: {row['score']:.3f}"
                    )
                    summary = recipe_summary(row)
                    if summary:
                        st.write(summary)
                    render_chips(metadata_tags(row) + row.get("vibe_tags", [])[:3])
                with right:
                    if st.button("View Recipe", key=f"view_ranked_{row['record_id']}"):
                        st.session_state["browser_search"] = row["record_id"]
                        st.rerun()


def render_recommendation_tab(df):
    """Render global cookbook recommendations with submit-only LLM calls."""
    st.subheader("Ask the cookbook")
    st.caption("Try a mood, season, ingredient, or occasion. The app searches your saved recipes.")

    examples = [
        "warming recipes for a cold day",
        "bright spring dinner in the sun",
        "impressive but not too much work",
    ]
    render_chips(examples)

    with st.form("global_recommendation_form"):
        global_query = st.text_input(
            "What sounds good?",
            placeholder="Try: cozy rainy-day dinner, lemony summer lunch, easy dinner party",
            key="global_recommendation_query",
        )

        top_n = st.slider(
            "Recommendations",
            min_value=3,
            max_value=15,
            value=6,
            step=1,
            key="global_recommendation_top_n",
        )

        submitted_recommendation = st.form_submit_button("Find recipes")

    # Streamlit reruns the whole script often; only this submit branch may call
    # global retrieval or the recommendation LLM. Later reruns redraw saved results.
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

    if not last_recommendation:
        st.info("Ask for a mood, occasion, or ingredient and your saved cookbook will suggest matches.")
        return

    recommendation = last_recommendation.get("markdown", "")
    ranked_matches = last_recommendation.get("ranked_matches")
    last_query = last_recommendation.get("query", "")

    if last_query:
        st.caption(
            f"Last search: {last_query} "
            f"({last_recommendation.get('top_n')} requested)"
        )

    if recommendation:
        with st.container(border=True):
            st.markdown("#### Recommended from your cookbook")
            st.markdown(recommendation)
    else:
        st.warning("No matching recipes found.")

    if ranked_matches is not None and not ranked_matches.empty:
        render_ranked_matches(ranked_matches)


def render_browse_tab(df, favorites_only):
    """Render searchable recipe browser."""
    search = st.text_input(
        "Search recipes",
        placeholder="Try: aubergine, chickpeas, cozy, lemon, brunch",
        key="browser_search",
    )
    filtered = filter_recipe_dataframe(df, search, favorites_only)

    st.write(f"Showing **{len(filtered)}** results.")

    if filtered.empty:
        st.info("No recipes match that search yet. Try a broader ingredient, mood, or page id.")
        return

    filtered = filtered.sort_values("title")
    for _, row in filtered.iterrows():
        render_recipe_card(row, key_prefix="browse")


def render_favorites_tab(df):
    """Render favorite recipes without changing rating/favorite logic."""
    favorites = df[df["favorite"]].sort_values("title")

    if favorites.empty:
        st.info("No favorites yet. Recipes become favorites when notes sound enthusiastic or rating is 4+.")
        return

    st.write(f"Showing **{len(favorites)}** favorites.")
    for _, row in favorites.iterrows():
        render_recipe_card(row, key_prefix="favorite")


def render_add_recipe_tab():
    """Render upload/OCR workflow."""
    st.subheader("Add a recipe")
    st.caption("Upload a clear photo from your phone or browser. OCR and metadata extraction may take a moment.")

    uploaded_image = st.file_uploader(
        "Upload recipe photo",
        type=["jpg", "jpeg", "png"],
        key="upload_recipe_photo",
    )

    if uploaded_image is not None:
        st.write(f"Ready to add: **{uploaded_image.name}**")

        if st.button("OCR and add recipe"):
            image_bytes = uploaded_image.getvalue()
            image_name = uploaded_image.name

            with st.status("Running OCR and metadata extraction...", expanded=False):
                metadata = ingest_uploaded_recipe(image_bytes, image_name)

            st.success(f"Added: {metadata.get('title', image_name)}")
            st.session_state["refresh_search_index_after_upload"] = True
            st.rerun()
    else:
        st.info("No photo selected yet.")


add_page_styles()

if "browser_search" not in st.session_state:
    # Browser filtering is independent from global recommendation form state.
    st.session_state["browser_search"] = st.session_state.get("recipe_browser_search", "")

st.sidebar.header("Filters")
favorites_only = st.sidebar.checkbox("Favorites only", value=False)
show_hidden = st.sidebar.checkbox("Show continuation/partial pages", value=False)

df = load_recipe_dataframe(DATASETS)

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

favorite_count = int(df["favorite"].sum()) if "favorite" in df.columns else 0
dataset_counts = df["dataset"].value_counts().sort_index()

st.markdown(
    """
    <div class="hero">
        <h1>RecipeBox</h1>
        <p>Rediscover recipes you already loved enough to save.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(3)
metric_cols[0].metric("Recipe pages", len(df))
metric_cols[1].metric("Favorites", favorite_count)
metric_cols[2].metric("Datasets", len(dataset_counts))
st.caption(
    "Loaded from "
    + ", ".join(f"{name} ({count})" for name, count in dataset_counts.items())
)

ask_tab, browse_tab, favorites_tab, add_tab = st.tabs([
    "Ask Cookbook",
    "Browse",
    "Favorites",
    "Add Recipe",
])

with ask_tab:
    render_recommendation_tab(df)

with browse_tab:
    render_browse_tab(df, favorites_only)

with favorites_tab:
    render_favorites_tab(df)

with add_tab:
    render_add_recipe_tab()
