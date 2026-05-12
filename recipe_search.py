import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from recipe_storage import format_user_notes, load_ocr_text_for_source

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_VERSION = 1


def format_tags(tags):
    """Render list metadata tags for retrieval text."""
    return ", ".join(tags) if isinstance(tags, list) else ""


def cosine_similarity(a, b):
    """Compare two embedding vectors with cosine similarity."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def text_hash(text):
    """Hash searchable text so stale cached embeddings can be detected."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expand_query(query):
    """Generate alternate phrasings for broader semantic retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 alternative phrasings of the query that preserve meaning.",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    lines = response.choices[0].message.content.split("\n")
    return [q.strip("- ").strip() for q in lines if q.strip()]


def build_recipe_search_text(row, ocr_text=None):
    """Combine metadata and OCR text into one retrieval document."""
    if ocr_text is None:
        ocr_text = load_ocr_text_for_source(
            row["source_image"],
            row.get("ocr_dir", "ocr_pages"),
        )

    parts = [
        f"Dataset: {row.get('dataset', '')}",
        f"Title: {row.get('title', '')}",
        f"Dish type: {row.get('dish_type', '')}",
        f"Main ingredients: {', '.join(row.get('main_ingredients', []))}",
        f"Description: {row.get('short_description', '')}",
        f"Semantic summary: {row.get('semantic_summary', '')}",
        f"Vibe tags: {format_tags(row.get('vibe_tags', []))}",
        f"Season tags: {format_tags(row.get('season_tags', []))}",
        f"Meal context tags: {format_tags(row.get('meal_context_tags', []))}",
        f"Effort level: {row.get('effort_level', '')}",
        f"Served temperature: {row.get('served_temperature', '')}",
        f"Make ahead potential: {row.get('make_ahead_potential', '')}",
        f"User notes: {format_user_notes(row.get('user_notes', ''))}",
        f"Recipe text: {ocr_text[:2500]}",
    ]

    return "\n".join(parts)


def _recipe_cache_records(df):
    """Build cache metadata records for every recipe/page row."""
    records = []
    for _, row in df.iterrows():
        search_text = build_recipe_search_text(row)
        records.append({
            "record_id": row.get("record_id", row["source_image"]),
            "source_image": row["source_image"],
            "dataset": row.get("dataset", ""),
            "ocr_dir": row.get("ocr_dir", "ocr_pages"),
            "title": row["title"],
            "search_text": search_text,
            "text_hash": text_hash(search_text),
        })
    return records


def _cache_is_valid(cache, records):
    """Check that the embedding cache matches the current recipe text."""
    if cache.get("version") != CACHE_VERSION:
        return False
    if cache.get("model") != EMBEDDING_MODEL:
        return False
    cached = cache.get("records", [])
    if len(cached) != len(records):
        return False

    current = {
        record["record_id"]: record["text_hash"]
        for record in records
    }
    cached_hashes = {
        record.get("record_id", record.get("source_image")): record.get("text_hash")
        for record in cached
    }
    return current == cached_hashes


def _load_embedding_cache(cache_path):
    """Load an embedding cache, returning an empty cache if unreadable."""
    if not cache_path.exists():
        return {"version": CACHE_VERSION, "model": EMBEDDING_MODEL, "records": []}

    try:
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return {"version": CACHE_VERSION, "model": EMBEDDING_MODEL, "records": []}

    if cache.get("version") != CACHE_VERSION or cache.get("model") != EMBEDDING_MODEL:
        return {"version": CACHE_VERSION, "model": EMBEDDING_MODEL, "records": []}

    return cache


def _save_embedding_cache(cache_path, records):
    """Persist embedding cache records to disk."""
    cache = {
        "version": CACHE_VERSION,
        "model": EMBEDDING_MODEL,
        "records": records,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)


def _refresh_recipe_embedding_cache(df, cache_path, force=False):
    """Incrementally refresh cache records and return records plus stats."""
    cache_path = Path(cache_path)
    records = _recipe_cache_records(df)
    cache = _load_embedding_cache(cache_path)
    cached_by_source = {
        record.get("record_id", record.get("source_image")): record
        for record in cache.get("records", [])
    }

    current_sources = {record["record_id"] for record in records}
    stale_sources = set(cached_by_source) - current_sources
    refreshed_records = []
    stats = {
        "reused": 0,
        "generated": 0,
        "removed_stale": len(stale_sources),
        "total": len(records),
    }

    print(f"Refreshing recipe embedding cache: {cache_path}")
    for record in records:
        cached = cached_by_source.get(record["record_id"])
        if (
            cached
            and not force
            and cached.get("text_hash") == record["text_hash"]
            and cached.get("embedding") is not None
        ):
            refreshed_records.append({
                **record,
                "embedding": cached["embedding"],
            })
            stats["reused"] += 1
            continue

        print(f"Embedding recipe: {record['record_id']}")
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=record["search_text"],
        ).data[0].embedding
        refreshed_records.append({
            **record,
            "embedding": embedding,
        })
        stats["generated"] += 1

    _save_embedding_cache(cache_path, refreshed_records)
    print(
        "Embedding cache refreshed: "
        f"{stats['reused']} reused, "
        f"{stats['generated']} generated, "
        f"{stats['removed_stale']} removed, "
        f"{stats['total']} total"
    )
    return refreshed_records, stats


def refresh_recipe_embedding_cache(df, cache_path="recipe_embedding_cache.pkl", force=False):
    """Incrementally refresh recipe embeddings and return summary stats."""
    _, stats = _refresh_recipe_embedding_cache(df, cache_path, force=force)
    return stats


def build_or_load_recipe_embeddings(df, cache_path="recipe_embedding_cache.pkl", force=False):
    """Load cached embeddings, incrementally generating missing or stale rows."""
    records = _recipe_cache_records(df)
    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        cache = _load_embedding_cache(cache_path)
        if _cache_is_valid(cache, records):
            return cache["records"]

    refreshed_records, _ = _refresh_recipe_embedding_cache(df, cache_path, force=force)
    return refreshed_records


def retrieve_ranked_recipes(query, df, top_k=30, cache_path="recipe_embedding_cache.pkl"):
    """Rank recipes with cached page embeddings and query-time local scoring."""
    recipe_embeddings = build_or_load_recipe_embeddings(df, cache_path=cache_path)
    if not recipe_embeddings:
        return pd.DataFrame()

    expanded_queries = [query] + expand_query(query)

    all_scores = []

    for q in expanded_queries:
        q_emb = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=q,
        ).data[0].embedding

        for record in recipe_embeddings:
            score = cosine_similarity(q_emb, record["embedding"])
            all_scores.append((score, record))

    # Keep best score per page.
    best = {}

    for score, record in all_scores:
        source = record["record_id"]

        if source not in best or score > best[source][0]:
            best[source] = (score, record)

    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
    rank_by_source = {
        record["record_id"]: {
            "rank": index,
            "score": score,
        }
        for index, (score, record) in enumerate(ranked[:top_k], start=1)
    }

    rows = []
    for _, row in df.iterrows():
        source = row.get("record_id", row["source_image"])
        if source not in rank_by_source:
            continue

        ranked_row = row.to_dict()
        ranked_row.update(rank_by_source[source])
        rows.append(ranked_row)

    return pd.DataFrame(rows).sort_values("rank")


def retrieve_recipe_candidates(user_query, df, top_k=8):
    """Return ranked candidate dictionaries for older callers."""
    ranked_df = retrieve_ranked_recipes(user_query, df, top_k=top_k)
    return [
        {
            "source_image": row["source_image"],
            "record_id": row.get("record_id", row["source_image"]),
            "title": row["title"],
            "row": row,
            "score": row["score"],
        }
        for _, row in ranked_df.iterrows()
    ]


def recommend_from_cookbook(user_query, ranked_candidates, top_n=6):
    """Recommend cookbook recipes using only retrieved local candidates."""
    max_candidates = min(12, top_n * 2)
    candidates = ranked_candidates.head(max_candidates)

    candidate_blocks = []

    for _, row in candidates.iterrows():
        source = row["source_image"]
        ocr_text = load_ocr_text_for_source(source, row.get("ocr_dir", "ocr_pages"))

        candidate_blocks.append(
            f"""
Page: {source}
Dataset: {row.get("dataset", "")}
Title: {row['title']}
Dish type: {row['dish_type']}
Main ingredients: {", ".join(row["main_ingredients"])}
Description: {row["short_description"]}
Semantic summary: {row.get("semantic_summary", "")}
Vibe tags: {format_tags(row.get("vibe_tags", []))}
Season tags: {format_tags(row.get("season_tags", []))}
Meal context tags: {format_tags(row.get("meal_context_tags", []))}
Effort level: {row.get("effort_level", "")}
Served temperature: {row.get("served_temperature", "")}
Make ahead potential: {row.get("make_ahead_potential", "")}
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
                ),
            },
            {
                "role": "user",
                "content": f"Cookbook candidates:\n{context}\n\nRequest:\n{user_query}",
            },
        ],
    )

    return response.choices[0].message.content


def answer_recipe_question(recipe_text, question):
    """Answer one-off questions about a single OCR recipe."""
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

    return response.choices[0].message.content
