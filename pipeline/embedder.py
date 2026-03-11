"""Embedding pipeline for text chunks and page images.

Uses Gemini Embedding 2 (gemini-embedding-2-preview) for both text and
image embeddings in a single semantic space. Also generates TF-IDF sparse
vectors for hybrid search on the text index.
"""

import json
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
from google.genai import types
from google.cloud import storage

from src.config import (
    PROJECT_ID, CREDENTIALS, BUCKET_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    TEXT_EMBED_BATCH_SIZE, IMAGE_EMBED_WORKERS,
    SPARSE_VECTORIZER_PATH,
)


def _get_genai_client() -> genai.Client:
    return genai.Client(
        vertexai=True, project=PROJECT_ID,
        location="us-central1", credentials=CREDENTIALS,
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)


# -- Text Embeddings ----------------------------------------------------------

def embed_text_chunks(chunks: List[Dict]) -> List[Dict]:
    """Generate dense embeddings for text chunks in parallel.

    Vertex AI embed_content only supports one content at a time,
    so we parallelize with ThreadPoolExecutor.
    """
    client = _get_genai_client()

    def embed_single(text: str) -> List[float]:
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                return response.embeddings[0].values
            except Exception as e:
                if attempt < 2 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                    time.sleep(2 ** attempt * 10)
                    continue
                raise

    results = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(embed_single, c["text"]): i
            for i, c in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Failed to embed text chunk {idx}: {e}")
                results[idx] = None

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = results.get(i)

    embedded = sum(1 for c in chunks if c.get("embedding"))
    print(f"Embedded {embedded}/{len(chunks)} text chunks")
    return chunks


# -- Image Embeddings ---------------------------------------------------------

def embed_page_images(page_images: List[Dict]) -> List[Dict]:
    """Generate embeddings for page images in parallel.

    Uses 3 workers to stay under the embedding token quota.
    """
    client = _get_genai_client()
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(BUCKET_NAME)

    def embed_single(page: Dict, max_retries: int = 3) -> Tuple[Dict, List[float]]:
        blob_path = page["gcs_uri"].replace(f"gs://{BUCKET_NAME}/", "")
        png_bytes = bucket.blob(blob_path).download_as_bytes()

        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png")],
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                return page, response.embeddings[0].values
            except Exception as e:
                if attempt < max_retries - 1 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                    time.sleep(2 ** attempt * 10)
                    continue
                raise

    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(embed_single, p): i
            for i, p in enumerate(page_images)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                page, embedding = future.result()
                results[idx] = embedding
            except Exception as e:
                print(f"Failed to embed page image {idx}: {e}")
                results[idx] = None

    for i, page in enumerate(page_images):
        page["embedding"] = results.get(i)

    embedded_count = sum(1 for p in page_images if p["embedding"] is not None)
    print(f"Embedded {embedded_count}/{len(page_images)} page images")
    return page_images


# -- Sparse (TF-IDF) Embeddings -----------------------------------------------

def build_sparse_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Fit a TF-IDF vectorizer on the corpus and generate sparse embeddings.

    Saves the vectorizer state to GCS so it can be loaded at query time.

    Args:
        chunks: List of chunk dicts, each must have a "text" key.

    Returns:
        Same list with a "sparse_embedding" key added to each dict.
    """
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    for i, chunk in enumerate(chunks):
        row = tfidf_matrix.getrow(i)
        values = [float(v) for v in row.data]
        dimensions = [int(d) for d in row.indices]
        chunk["sparse_embedding"] = {"values": values, "dimensions": dimensions}

    # Persist vectorizer state to GCS for query-time use
    _save_vectorizer_state(vectorizer)
    print(f"Built sparse embeddings for {len(chunks)} chunks, vocab size: {len(vectorizer.vocabulary_)}")

    return chunks


def _save_vectorizer_state(vectorizer: TfidfVectorizer):
    """Serialize TF-IDF vectorizer state and upload to GCS."""
    state = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "ngram_range": list(vectorizer.ngram_range),
    }

    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(SPARSE_VECTORIZER_PATH)
    blob.upload_from_string(
        json.dumps(state), content_type="application/json"
    )
    print(f"Saved vectorizer state to gs://{BUCKET_NAME}/{SPARSE_VECTORIZER_PATH}")
