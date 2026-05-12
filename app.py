from html import escape
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from recipe_ingest import extract_recipe_metadata, ingest_uploaded_recipe
from recipe_search import (
    answer_recipe_question,
    generate_inspired_recipe,
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
    {
        "name": "generated",
        "ocr_dir": "generated_recipes",
        "metadata_dir": "generated_recipe_metadata",
    },
]

GENERATED_OCR_DIR = "generated_recipes"
GENERATED_METADATA_DIR = "generated_recipe_metadata"
DATASETS_KEY = tuple(
    (
        dataset["name"],
        dataset["ocr_dir"],
        dataset["metadata_dir"],
        dataset.get("review_csv", ""),
    )
    for dataset in DATASETS
)


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


@st.cache_data(show_spinner=False, ttl=30)
def cached_recipe_dataframe(datasets_key):
    """Cache metadata loading briefly so ordinary UI clicks stay snappy."""
    datasets = [
        {
            "name": name,
            "ocr_dir": ocr_dir,
            "metadata_dir": metadata_dir,
            **({"review_csv": review_csv} if review_csv else {}),
        }
        for name, ocr_dir, metadata_dir, review_csv in datasets_key
    ]
    return load_recipe_dataframe(datasets)


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


def order_matches_like_recommendations(recommendation, ranked_matches):
    """Put cards in the same order as the AI recommendation text when possible."""
    if not recommendation or ranked_matches is None or ranked_matches.empty:
        return ranked_matches

    recommendation_text = recommendation.lower()
    matched = []
    unmatched = []

    for _, row in ranked_matches.iterrows():
        markers = [
            str(row.get("record_id", "")),
            str(row.get("source_image", "")),
            str(row.get("title", "")),
        ]
        positions = [
            recommendation_text.find(marker.lower())
            for marker in markers
            if marker and recommendation_text.find(marker.lower()) >= 0
        ]

        if positions:
            matched.append((min(positions), row))
        else:
            unmatched.append(row)

    if not matched:
        return ranked_matches

    ordered_rows = [row for _, row in sorted(matched, key=lambda item: item[0])]
    ordered_rows.extend(unmatched)
    return pd.DataFrame(ordered_rows)


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
            load_text_key = f"load_text_{key_prefix}_{row['record_id']}"
            if st.button("Load recipe text", key=load_text_key):
                st.session_state[load_text_key] = True

            if st.session_state.get(load_text_key) and ocr_path.exists():
                st.text(ocr_path.read_text(encoding="utf-8"))
            elif st.session_state.get(load_text_key):
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


def selected_inspiration_df(df, selected_ids):
    """Return selected inspiration recipes in UI-selected order."""
    by_id = {row["record_id"]: row for _, row in df.iterrows()}
    rows = [by_id[record_id] for record_id in selected_ids if record_id in by_id]
    return pd.DataFrame(rows)


def recipe_option_labels(df):
    """Build stable labels for Create tab multiselect choices."""
    return {
        row["record_id"]: f"{row['title']} ({row['dataset']} | {row['source_image']})"
        for _, row in df.iterrows()
    }


def combine_selected_ids(*groups):
    """Merge selected recipe ids while preserving selection order."""
    selected = []
    seen = set()
    for group in groups:
        for record_id in group:
            if record_id not in seen:
                selected.append(record_id)
                seen.add(record_id)
    return selected


def render_inspiration_preview_card(row):
    """Render compact inspiration context without action buttons."""
    with st.container(border=True):
        st.markdown(f"**{row['title']}**")
        st.caption(f"{row['dataset']} | {row['source_image']}")
        summary = recipe_summary(row)
        if summary:
            st.write(summary[:260])
        if row.get("main_ingredients"):
            render_chips(row["main_ingredients"][:8])


def save_generated_recipe(recipe_text, source_hint="generated_recipe"):
    """Save generated recipe text and metadata into the generated dataset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", source_hint.lower()).strip("_")[:48] or "recipe"
    digest = hashlib.sha1(recipe_text.encode("utf-8")).hexdigest()[:8]
    stem = f"generated_{slug}_{timestamp}_{digest}"
    source_image = f"{stem}.txt"

    ocr_dir = Path(GENERATED_OCR_DIR)
    metadata_dir = Path(GENERATED_METADATA_DIR)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    txt_path = ocr_dir / f"{stem}.txt"
    json_path = metadata_dir / f"{stem}.json"

    if txt_path.exists() or json_path.exists():
        raise FileExistsError(f"Generated recipe already exists for {stem}")

    metadata = extract_recipe_metadata(
        recipe_text,
        source_image=source_image,
        raw_output_path=metadata_dir / f"{stem}.raw.txt",
    )
    metadata["source_image"] = source_image
    metadata["dataset"] = "generated"
    metadata["dish_type"] = metadata.get("dish_type") or "generated"
    if not isinstance(metadata.get("user_notes"), list):
        metadata["user_notes"] = []
    metadata["user_notes"].append("AI-generated recipe")
    txt_path.write_text(recipe_text, encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return txt_path, json_path


def render_ranked_matches(ranked_matches, recommendation=""):
    """Render compact ranked recommendation cards."""
    ordered_matches = order_matches_like_recommendations(recommendation, ranked_matches)

    st.subheader("All ranked matches")
    with st.container(height=430, border=True):
        for display_rank, (_, row) in enumerate(ordered_matches.iterrows(), start=1):
            with st.container(border=True):
                left, right = st.columns([0.78, 0.22])
                with left:
                    st.markdown(f"**{display_rank}. {row['title']}**")
                    st.caption(
                        f"Dataset: {row['dataset']} | "
                        f"Page: {row['source_image']} | "
                        f"Score: {row['score']:.3f} | "
                        f"Semantic rank: {int(row['rank'])}"
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
        render_ranked_matches(ranked_matches, recommendation)


def render_browse_tab(df, favorites_only):
    """Render searchable recipe browser."""
    search = st.text_input(
        "Search recipes",
        placeholder="Try: aubergine, chickpeas, cozy, lemon, brunch",
        key="browser_search",
    )
    search_mode_label = st.selectbox(
        "Search mode",
        options=[
            "All fields",
            "Ingredients: broad",
            "Ingredients: exact",
        ],
        index=0,
        help=(
            "Use exact ingredient search when you want 'corn' to match corn "
            "but not corn flour."
        ),
    )
    search_mode = {
        "All fields": "all",
        "Ingredients: broad": "ingredients_broad",
        "Ingredients: exact": "ingredients_exact",
    }[search_mode_label]

    filtered = filter_recipe_dataframe(df, search, favorites_only, search_mode=search_mode)

    st.write(f"Showing **{len(filtered)}** results.")

    if filtered.empty:
        st.info("No recipes match that search yet. Try a broader ingredient, mood, or page id.")
        return

    filtered = filtered.sort_values("title")
    max_cards = st.slider(
        "Cards to show",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        key="browse_card_limit",
    )
    if len(filtered) > max_cards:
        st.caption(f"Showing the first {max_cards} cards. Search more specifically to narrow results.")

    for _, row in filtered.head(max_cards).iterrows():
        render_recipe_card(row, key_prefix="browse")


def render_favorites_tab(df):
    """Render favorite recipes without changing rating/favorite logic."""
    favorites = df[df["favorite"]].sort_values("title")

    if favorites.empty:
        st.info("No favorites yet. Recipes become favorites when notes sound enthusiastic or rating is 4+.")
        return

    max_cards = st.slider(
        "Favorite cards to show",
        min_value=10,
        max_value=100,
        value=min(30, max(10, len(favorites))),
        step=10,
        key="favorite_card_limit",
    )

    st.write(f"Showing **{min(len(favorites), max_cards)}** of **{len(favorites)}** favorites.")
    for _, row in favorites.head(max_cards).iterrows():
        render_recipe_card(row, key_prefix="favorite")


def render_create_tab(df):
    """Render AI recipe creation grounded in selected saved recipes."""
    st.subheader("Create from your saved cookbook")
    st.caption(
        "Generate a new AI recipe idea inspired by recipes you already saved. "
        "Pick inspiration recipes first, then add what you have and the mood you want."
    )

    left_col, right_col = st.columns(2)
    search_selected_ids = []
    semantic_selected_ids = []

    with left_col:
        st.markdown("#### Find recipes to inspire it")
        create_search = st.text_input(
            "Search saved recipes",
            placeholder="Try: corn, lentils, bright salad, cozy soup",
            key="create_search",
        )
        if create_search:
            matches = filter_recipe_dataframe(df, create_search, favorites_only=False).head(12)
            if matches.empty:
                st.info("No saved recipes match that search.")
            else:
                options = matches["record_id"].tolist()
                labels = recipe_option_labels(matches)
                current = [
                    record_id
                    for record_id in st.session_state.get("create_search_inspiration_ids", [])
                    if record_id in options
                ]
                st.session_state["create_search_inspiration_ids"] = current
                search_selected_ids = st.multiselect(
                    "Choose search results as inspiration",
                    options=options,
                    format_func=lambda record_id: labels.get(record_id, record_id),
                    key="create_search_inspiration_ids",
                )
                for _, row in matches.head(6).iterrows():
                    render_inspiration_preview_card(row)

    with right_col:
        st.markdown("#### Semantic inspiration search")
        semantic_prompt = st.text_input(
            "Retrieve by mood, style, or context",
            placeholder="Try: bright Ottolenghi-style salads, cozy lentil dinners",
            key="create_semantic_prompt",
        )
        semantic_top_n = st.slider(
            "Candidates",
            min_value=3,
            max_value=12,
            value=6,
            key="create_semantic_top_n",
        )
        if st.button("Find inspiration candidates"):
            if semantic_prompt.strip():
                with st.spinner("Retrieving inspiration candidates..."):
                    st.session_state["create_semantic_candidates"] = cached_ranked_recipes(
                        semantic_prompt.strip(),
                        semantic_top_n,
                        dataframe_cache_key(df),
                        df,
                    )
            else:
                st.warning("Enter a semantic inspiration prompt first.")

        candidates = st.session_state.get("create_semantic_candidates")
        if candidates is not None and not candidates.empty:
            candidate_options = candidates.head(semantic_top_n)["record_id"].tolist()
            candidate_labels = recipe_option_labels(candidates.head(semantic_top_n))
            current = [
                record_id
                for record_id in st.session_state.get("create_semantic_inspiration_ids", [])
                if record_id in candidate_options
            ]
            st.session_state["create_semantic_inspiration_ids"] = current
            semantic_selected_ids = st.multiselect(
                "Choose retrieved candidates as inspiration",
                options=candidate_options,
                format_func=lambda record_id: candidate_labels.get(record_id, record_id),
                key="create_semantic_inspiration_ids",
            )
            for _, row in candidates.head(min(6, semantic_top_n)).iterrows():
                render_inspiration_preview_card(row)

    st.divider()
    selected_ids = combine_selected_ids(search_selected_ids, semantic_selected_ids)
    selected_df = selected_inspiration_df(df, selected_ids)

    st.markdown("#### Selected inspiration recipes")
    if selected_df.empty:
        st.info("No inspiration recipes selected yet. Pick recipes above, then click Generate.")
    else:
        for _, row in selected_df.iterrows():
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['dataset']} | {row['source_image']}")

    st.divider()
    st.markdown("#### Shape the new idea")

    ingredients_on_hand = st.text_area(
        "Ingredients on hand",
        placeholder="e.g. corn, tomatoes, herbs, Greek yogurt",
        key="create_ingredients_on_hand",
    )
    ingredients_to_use_or_avoid = st.text_area(
        "Ingredients to use or avoid",
        placeholder="e.g. use lemons; avoid shellfish; no cilantro",
        key="create_use_avoid",
    )
    desired_mood_or_context = st.text_input(
        "Desired mood or context",
        placeholder="e.g. sunny lunch, cozy weeknight, casual dinner with friends",
        key="create_mood_context",
    )
    dietary_constraints = st.text_input(
        "Dietary constraints",
        placeholder="e.g. vegetarian, gluten-free, dairy-light",
        key="create_dietary",
    )
    time_effort_preference = st.text_input(
        "Time / effort preference",
        placeholder="e.g. under 45 minutes, low effort, impressive but manageable",
        key="create_effort",
    )

    can_generate = not selected_df.empty or any([
        ingredients_on_hand.strip(),
        desired_mood_or_context.strip(),
        ingredients_to_use_or_avoid.strip(),
    ])

    if st.button("Create AI recipe idea", disabled=not can_generate):
        user_inputs = {
            "ingredients_on_hand": ingredients_on_hand,
            "ingredients_to_use_or_avoid": ingredients_to_use_or_avoid,
            "desired_mood_or_context": desired_mood_or_context,
            "dietary_constraints": dietary_constraints,
            "time_effort_preference": time_effort_preference,
        }
        with st.spinner("Creating a new recipe idea from your saved inspiration..."):
            generated_text = generate_inspired_recipe(selected_df, user_inputs)
        st.session_state["generated_recipe_text"] = generated_text

    generated_text = st.session_state.get("generated_recipe_text", "")
    if generated_text:
        st.markdown("#### AI-generated recipe idea")
        st.warning("This is AI-generated and inspired by your saved recipes, not copied from them.")
        st.markdown(generated_text)

        if st.button("Save generated recipe"):
            source_hint = desired_mood_or_context or ingredients_on_hand or "generated_recipe"
            with st.spinner("Saving generated recipe and extracting metadata..."):
                txt_path, json_path = save_generated_recipe(generated_text, source_hint)
                cached_recipe_dataframe.clear()
                updated_df = cached_recipe_dataframe(DATASETS_KEY)
                if not show_hidden:
                    updated_df = updated_df[~updated_df["hide_from_browse"]]
                stats = refresh_recipe_embedding_cache(updated_df, force=False)
                cached_ranked_recipes.clear()
            st.session_state["refresh_search_index_after_upload"] = False
            st.success(
                "Saved generated recipe: "
                f"{txt_path} and {json_path}. "
                f"Search index refreshed ({stats['generated']} generated)."
            )


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

df = cached_recipe_dataframe(DATASETS_KEY)

if not show_hidden:
    df = df[~df["hide_from_browse"]]

st.sidebar.header("Search index")

if st.session_state.pop("refresh_search_index_after_upload", False):
    cached_recipe_dataframe.clear()
    df = cached_recipe_dataframe(DATASETS_KEY)
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

section = st.radio(
    "Section",
    options=[
        "Ask Cookbook",
        "Browse",
        "Create",
        "Favorites",
        "Add Recipe",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# Streamlit tabs eagerly render every tab body. A single active section keeps
# ordinary clicks from rebuilding hundreds of hidden recipe cards.
if section == "Ask Cookbook":
    render_recommendation_tab(df)
elif section == "Browse":
    render_browse_tab(df, favorites_only)
elif section == "Create":
    render_create_tab(df)
elif section == "Favorites":
    render_favorites_tab(df)
else:
    render_add_recipe_tab()
