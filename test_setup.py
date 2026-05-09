# %%
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import re

load_dotenv() # loads the openai key
client = OpenAI()

VERBOSE=False

#==============================
# test data


documents = [

"""Whole-slide images (WSIs) are high-resolution digital scans of histopathology slides.
Deep learning models typically divide WSIs into tiles, extract features from each tile,
and aggregate these features to make slide-level predictions. These approaches have
shown promise in predicting clinical outcomes such as survival and recurrence.""",

"""The STAMP workflow is a computational pathology pipeline designed to analyze
whole-slide images for cancer diagnosis and prognosis. It uses deep learning to identify
morphological patterns associated with tumor subtype and recurrence risk. STAMP enables
scalable analysis of pathology slides without requiring manual annotation.""",

"""A key limitation of deep learning models in computational pathology is their lack
of interpretability. Although models may achieve high predictive performance, it is often
unclear which features of the tissue are driving predictions. Techniques such as attention
maps and saliency visualization are being explored to address this issue.""",

"""Whole genome sequencing provides a comprehensive view of genetic mutations in tumors.
In some cancer types, genomic features are highly predictive of treatment response and
disease progression. Genomic approaches can capture molecular alterations that are not
visible in histopathology images.""",

"""In contrast to genomic methods, pathology-based deep learning models can capture
spatial and morphological features of tumors. Some studies have shown that these models
can outperform genomic approaches in predicting recurrence, particularly when tissue
architecture is important.""",

"""Experts emphasize that both genomic and pathology-based approaches have complementary
strengths. The effectiveness of each method depends on the specific clinical task, cancer
type, and available data. Combining both modalities may lead to improved predictive
performance.""",

"""Histology slides are typically stained using hematoxylin and eosin (H&E), which
highlights cellular structures. Pathologists visually examine these slides to diagnose
disease. Digital pathology enables these slides to be scanned and analyzed computationally.""",

"""Machine learning models require large amounts of labeled data for training. In medical
imaging, obtaining high-quality annotations can be time-consuming and expensive. Weakly
supervised and self-supervised approaches are being developed to reduce annotation
requirements.""",

"""Recent work has explored predicting patient outcomes from pathology images using
convolutional neural networks and transformer-based architectures. While promising,
results vary depending on dataset size, quality, and preprocessing methods.""",

"""Some studies report high accuracy in predicting outcomes from pathology images,
but these results may not generalize across institutions due to differences in staining
protocols and scanner variability."""

]


test_queries = [
    {
        "query": "What is a limitation of computational pathology?",
        "expected_type": "answer",
        "must_contain": ["interpretability"],
        "expected_chunk_ids": [2]
    },
    {
        "query": "How are whole-slide images used?",
        "expected_type": "answer",
        "must_contain": ["tiles", "features"],
        "expected_chunk_ids": [0, 1]
    },
    {
        "query": "Is pathology better than genomics?",
        "expected_type": "nuanced",
        "expected_chunk_ids": [4, 5, 3]
    },
    {
        "query": "Which model is best for lung cancer?",
        "expected_type": "abstain",
        "expected_chunk_ids": []
    }
]

hard_test_queries = [
    {
        "query": "What evidence suggests pathology models may not transfer well between hospitals?",
        "expected_type": "answer",
        "must_contain": ["generalize", "staining", "scanner"],
        "expected_chunk_ids": [9]
    },
    {
        "query": "Why might combining pathology and genomics be useful?",
        "expected_type": "answer",
        "must_contain": ["complementary", "combining"],
        "expected_chunk_ids": [5]
    },
    {
        "query": "What approaches reduce the need for manual annotation?",
        "expected_type": "answer",
        "must_contain": ["weakly", "self-supervised"],
        "expected_chunk_ids": [7]
    },
    {
        "query": "Can pathology images reveal molecular alterations directly?",
        "expected_type": "nuanced",
        "expected_chunk_ids": [3, 4, 5]
    },
    {
        "query": "What is the best treatment for recurrence?",
        "expected_type": "abstain",
        "expected_chunk_ids": []
    }
]

test_queries = test_queries + hard_test_queries
#=====================================
#helpers


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# split long text into chunks
def chunk_text(text, max_words=120, overlap=30):
    sentences = split_sentences(text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))

            # overlap (keep last N words)
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words.copy()
            current_len = len(current_chunk)

        current_chunk.extend(words)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# compare 2 embeddings
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ecode docs into embeddings
def build_index(documents):
    all_chunks = []

    for doc_id, doc in enumerate(documents):

        text = doc["text"]
        source = doc["source"]
        all_chunks.extend(chunk_text(text))

    embeddings = []

    for chunk in all_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        embeddings.append({
            "id": len(embeddings),
            "text": chunk,
            "embedding": response.data[0].embedding,
            "doc_id" :doc_id, 
            "source" : source,

        })

    return embeddings

# encode a query, match it to the provided embeddings via cosine similarity
def retrieve(query, embeddings, top_k=5):
    queries = [query] + expand_query(query)

    all_scores = []

    for q in queries:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=q
        ).data[0].embedding

        for item in embeddings:
            score = cosine_similarity(q_emb, item["embedding"])
            all_scores.append((
                score,
                item["id"],
                item["text"],
                item["source"]
            ))

    # dedupe: keep best score per chunk id
    best = {}

    for score, chunk_id, text, source in all_scores:
        if chunk_id not in best or score > best[chunk_id][0]:
            best[chunk_id] = (score, text, source)

    scored = [
        (score, chunk_id, text, source)
        for chunk_id, (score, text, source) in best.items()
    ]

    scored.sort(reverse=True, key=lambda x: x[0])

    if VERBOSE:
        print("\nExpanded queries:")
        for q in queries:
            print("-", q)

        print("\nTop scores:")
        for s, i, t, source in scored[:top_k]:
            print(f"{s:.3f} | Chunk {i} | {source} | {t[:80]}")

    return [(i, t, source) for s, i, t, source in scored[:top_k]]


def rerank(query, chunks):

    formatted = "\n\n".join(
        [f"[Chunk {i}] {text}" for i, text in chunks]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Select the most relevant chunks for answering the question. "
                    "Return ONLY the chunk IDs in order of relevance, like: 2, 5, 1"
                )
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nChunks:\n{formatted}"
            }
        ]
    )

    text = response.choices[0].message.content

    # robust parsing
    import re
    ids = [int(x) for x in re.findall(r"\d+", text)]

    return ids

def answer_assistant(query, context_chunks):
    context = "\n\n".join(
        [
            f"[Chunk {chunk_id} | Page: {source}]\n{text}"
            for chunk_id, text, source in context_chunks
        ]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful cooking assistant. "
                    "Use the provided recipe context, but you may also use general cooking knowledge "
                    "to provide practical substitutions, prep advice, and cooking suggestions. "
                    "Clearly distinguish between recipe facts and general suggestions. "
                    "If referencing where recipe information came from, use ONLY the Page value "
                    "provided in the context. "
                    "Do not invent cookbook names, websites, publishers, or sources. "
                    "Keep answers concise and practical."
                )
                
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            },
        ]
    )

    return response.choices[0].message.content


# strict RAG approach. Answers only using context provided. Otherwise reports it cannot.
def answer(query, context_chunks):

    context = "\n\n".join(
        [f"[Chunk {chunk_id}]\n{text}" for chunk_id, text in context_chunks]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided chunks. "
                    "Cite chunk numbers like [Chunk 2]. "
                    "Every claim must have a citation. "
                    "If the context directly addresses the question, answer cautiously. "
                    "If the context does not support a definitive answer, explain what can "
                    "and cannot be concluded. "
                    "If the context is unrelated, say there is not enough evidence."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response.choices[0].message.content

def expand_query(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 alternative phrasings of the query that preserve meaning."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    lines = response.choices[0].message.content.split("\n")
    return [q.strip("- ").strip() for q in lines if q.strip()]

def run_query(query, index):
    retrieved = retrieve(query, index, top_k=5)

    if not retrieved:
        return "NO_CONTEXT"

    ranked_ids = rerank(query, retrieved)
    print("Reranked IDs:", ranked_ids)

    chunk_by_id = {chunk_id: text for chunk_id, text in retrieved}

    reranked = [
        (chunk_id, chunk_by_id[chunk_id])
        for chunk_id in ranked_ids[:3]
        if chunk_id in chunk_by_id
    ]

    return answer(query, reranked)

def evaluate(index):

    hits = 0
    total = 0
    top1_hits = 0
    top3_hits = 0
    total = 0

    for t in test_queries:
        print("\n====================")
        print("Query:", t["query"])

        retrieved = retrieve(t["query"], index, top_k=5)
        retrieved_ids = [i for i, _ in retrieved]

        print("Retrieved chunk IDs:", retrieved_ids)

        expected_ids = t.get("expected_chunk_ids", [])

        if expected_ids:
            total += 1

            if retrieved_ids:
                if retrieved_ids[0] in expected_ids:
                    top1_hits += 1
                    print("✅ Top-1 hit")
                else:
                    print("❌ Top-1 miss")

                if any(e in retrieved_ids[:3] for e in expected_ids):
                    top3_hits += 1
                    print("✅ Top-3 hit")
                else:
                    print("❌ Top-3 miss")

        if expected_ids and retrieved_ids[0] not in expected_ids:
            print("🔎 Top-1 miss details")
            print("Expected:", expected_ids)
            print("Retrieved:", retrieved_ids)

        if not retrieved:
            result = "NO_CONTEXT"
        else:
            result = answer(t["query"], retrieved)
        print("Result:", result)

    # keep your existing answer evaluation logic here
    print("\n==== Retrieval Metrics ====")
    print(f"Top-1 accuracy: {top1_hits}/{total}")
    print(f"Top-3 accuracy: {top3_hits}/{total}")


import base64
from pathlib import Path

def ocr_image(image_path):
    image_path = Path(image_path)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type = "image/jpeg"  # fine for .jpg/.jpeg

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract the handwritten recipe clearly. "
                            "Preserve title, ingredients, and numbered steps. "
                            "Do not summarize."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                    },
                ],
            }
        ],
    )

    return response.output_text


#%%
#----------------------------------------
# OCR a folder of images
import json
import re
from pathlib import Path
from tqdm import tqdm

def extract_recipe_metadata(recipe_text, source_image=None):

    with open("review.md", "r", encoding="utf-8") as f:
        review_examples = f.read()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract metadata from OCR'd recipe text. "
                    "Learn from the reviewed examples below."
                    f"{review_examples}"
                    "You MUST include every requested field. "
                    "If uncertain, use null. "
                    "Do not invent ingredients not present in the text."
                    "Return ONLY valid JSON. "
                    "You MUST include every requested field. "
                    "If uncertain, use null. "
                    "Do not invent ingredients not present in the text."
                ),
            },
            {
                "role": "user",
                "content": f"""
                            Recipe text:
                            {recipe_text}

                            Return JSON with exactly these fields:

                            {{
                            "source_image": {json.dumps(source_image)},
                            "title": string or null,
                            "title_confidence": number between 0 and 1,
                            "recipe_structure": one of [
                                "single_complete_recipe",
                                "multiple_recipes",
                                "continuation_page",
                                "partial_recipe",
                                "unknown"
                            ],
                            "likely_continuation": true or false,
                            "ocr_quality_score": number between 0 and 1,
                            "dish_type": string or null,
                            "main_ingredients": list of strings,
                            "short_description": string or null,
                            "user_notes": list of strings
                            }}
                            """
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    metadata = json.loads(raw)

    # harden against missing fields
    metadata.setdefault("source_image", source_image)
    metadata.setdefault("title", None)
    metadata.setdefault("title_confidence", None)
    metadata.setdefault("recipe_structure", "unknown")
    metadata.setdefault("likely_continuation", False)
    metadata.setdefault("ocr_quality_score", None)
    metadata.setdefault("dish_type", None)
    metadata.setdefault("main_ingredients", [])
    metadata.setdefault("short_description", None)
    metadata.setdefault("user_notes", [])

    return metadata


def process_recipe_images(image_dir, limit=5):

    image_dir = Path(image_dir)

    image_paths = sorted(image_dir.glob("*.jpg"))[:limit]

    documents = []

    for image_path in tqdm(image_paths):

        print(f"\n📸 Processing: {image_path.name}")

        # OCR
        recipe_text = ocr_image(image_path)

        # save OCR text
        txt_path = Path("ocr_pages") / f"{image_path.stem}.txt"
        txt_path.write_text(recipe_text, encoding="utf-8")

        # metadata extraction
        metadata = extract_recipe_metadata(recipe_text, source_image=image_path.name)

        # save metadata
        json_path = Path("recipe_metadata") / f"{image_path.stem}.json"

        json_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(f"🏷️ Title: {metadata['title']}")

        documents.append(recipe_text)

    return documents

# %%
recipe_path = Path("/Users/richardcorbett/Downloads/Photos-67")
if __name__ == "__main__":
    documents = process_recipe_images(
        recipe_path,
        limit=100
    )


def load_recipe_documents(ocr_dir="ocr_pages"):

    documents = []

    txt_paths = sorted(Path(ocr_dir).glob("*.txt"))

    for path in txt_paths:

        text = path.read_text(encoding="utf-8")

        documents.append({
            "source": path.stem,
            "text": text
        })

    return documents

if __name__ == "__main__":
    documents = load_recipe_documents()

if __name__ == "__main__":
    index = build_index(documents)

# %%
def show_sources(context_chunks):
    sources = sorted(set(
        chunk[2] for chunk in context_chunks
    ))

    print("\n📚 Using recipes/pages:")
    for source in sources:
        print(f"- {source}")
# %%
if __name__ == "__main__":
    query = "Can you recommend 4 recipes to cook in a night?"

    chunks = retrieve(query, index, top_k=5)
    show_sources(chunks)
    print(answer_assistant(query, chunks))
# %%
