import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from recipe_storage import format_user_notes, load_ocr_text_for_source

load_dotenv()
client = OpenAI()


def cosine_similarity(a, b):
    """Compare two embedding vectors with cosine similarity."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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


def build_recipe_search_text(row):
    """Combine metadata and OCR text into one retrieval document."""
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
    """Rank recipe pages against a query using OpenAI embeddings."""
    expanded_queries = [user_query] + expand_query(user_query)

    recipe_texts = []

    for _, row in df.iterrows():
        recipe_texts.append({
            "source_image": row["source_image"],
            "title": row["title"],
            "search_text": build_recipe_search_text(row),
            "row": row,
        })

    recipe_embeddings = []

    for item in recipe_texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=item["search_text"],
        ).data[0].embedding

        recipe_embeddings.append({
            **item,
            "embedding": emb,
        })

    all_scores = []

    for q in expanded_queries:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=q,
        ).data[0].embedding

        for item in recipe_embeddings:
            score = cosine_similarity(q_emb, item["embedding"])
            all_scores.append((score, item))

    # Keep best score per page.
    best = {}

    for score, item in all_scores:
        source = item["source_image"]

        if source not in best or score > best[source][0]:
            best[source] = (score, item)

    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)

    return [item for score, item in ranked[:top_k]]


def recommend_from_cookbook(user_query, df, max_candidates=12, top_n=6):
    """Recommend cookbook recipes using only retrieved local candidates."""
    candidates = retrieve_recipe_candidates(
        user_query,
        df,
        top_k=max_candidates,
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
