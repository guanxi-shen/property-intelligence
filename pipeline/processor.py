"""Main orchestrator for the document processing pipeline.

Chains: Document AI parsing -> page rendering -> embedding -> indexing.
Caches parsed chunks and page lists locally so DocAI costs aren't repeated.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from google.cloud import storage

from src.config import (
    BUCKET_NAME, CREDENTIALS, PROJECT_ID,
    UPLOADS_PREFIX, PROCESSED_PREFIX,
)
from pipeline.document_ai import parse_pdf
from pipeline.page_converter import convert_pdf_to_pages
from pipeline.embedder import embed_text_chunks, embed_page_images, build_sparse_embeddings
from pipeline.indexer import update_text_index, update_multimodal_index

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)


def _list_pdfs(prefix: str) -> List[str]:
    """List all PDF GCS URIs under a given prefix."""
    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=prefix)
    return [
        f"gs://{BUCKET_NAME}/{b.name}"
        for b in blobs
        if b.name.lower().endswith(".pdf")
    ]


def _move_blob(src_path: str, dest_path: str):
    """Move a blob within the same bucket."""
    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    src_blob = bucket.blob(src_path)
    bucket.copy_blob(src_blob, bucket, dest_path)
    src_blob.delete()


def _cache_path(pdf_name: str, kind: str) -> Path:
    """Local cache file path for parsed data."""
    stem = Path(pdf_name).stem
    return CACHE_DIR / f"{stem}_{kind}.json"


def _save_cache(pdf_name: str, kind: str, data: List[Dict]):
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(pdf_name, kind)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Cached {kind}: {path.name}")


def _load_cache(pdf_name: str, kind: str) -> List[Dict]:
    path = _cache_path(pdf_name, kind)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded from cache: {path.name} ({len(data)} items)")
        return data
    return None


def process_document(gcs_uri: str) -> Dict:
    """Process a single PDF end-to-end: parse, render, embed, index."""
    pdf_name = gcs_uri.rstrip("/").split("/")[-1]
    print(f"Processing: {pdf_name}")

    # Step 1: Parse (use cache if available)
    chunks = _load_cache(pdf_name, "chunks")
    if chunks is None:
        print(f"  [1/4] Parsing with Layout Parser...")
        chunks = parse_pdf(gcs_uri)
        _save_cache(pdf_name, "chunks", chunks)
    print(f"         {len(chunks)} chunks")

    # Step 2: Render pages (use cache if available)
    page_images = _load_cache(pdf_name, "pages")
    if page_images is None:
        print(f"  [2/4] Rendering pages to PNG...")
        page_images = convert_pdf_to_pages(gcs_uri)
        _save_cache(pdf_name, "pages", page_images)
    print(f"         {len(page_images)} pages")

    # Step 3: Embed
    print(f"  [3/4] Generating embeddings...")
    chunks = embed_text_chunks(chunks)
    chunks = build_sparse_embeddings(chunks)
    page_images = embed_page_images(page_images)

    # Step 4: Index
    print(f"  [4/4] Updating indexes...")
    update_text_index(chunks)
    update_multimodal_index(page_images)

    print(f"Done: {pdf_name}")
    return {"source_pdf": pdf_name, "chunks": len(chunks), "pages": len(page_images)}


def process_all_new():
    """Scan uploads/, process all new PDFs with local caching.

    Parsing and page rendering results are cached to disk.
    If a PDF was previously parsed, the cache is reused.
    """
    gcs_uris = _list_pdfs(UPLOADS_PREFIX)
    if not gcs_uris:
        print("No new PDFs found in uploads/")
        return []

    print(f"Found {len(gcs_uris)} new PDFs to process")

    all_chunks = []
    all_pages = []

    for uri in gcs_uris:
        pdf_name = uri.rstrip("/").split("/")[-1]
        print(f"\n--- {pdf_name} ---")

        # Parse with cache
        chunks = _load_cache(pdf_name, "chunks")
        if chunks is None:
            try:
                chunks = parse_pdf(uri)
                _save_cache(pdf_name, "chunks", chunks)
            except Exception as e:
                print(f"  PARSE ERROR: {e}")
                chunks = []
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

        # Render with cache
        page_images = _load_cache(pdf_name, "pages")
        if page_images is None:
            try:
                page_images = convert_pdf_to_pages(uri)
                _save_cache(pdf_name, "pages", page_images)
            except Exception as e:
                print(f"  RENDER ERROR: {e}")
                page_images = []
        print(f"  {len(page_images)} pages")
        all_pages.extend(page_images)

    print(f"\n--- Totals: {len(all_chunks)} chunks, {len(all_pages)} pages ---")

    # Embed text
    print(f"\nEmbedding {len(all_chunks)} text chunks...")
    all_chunks = embed_text_chunks(all_chunks)

    print(f"Building sparse embeddings...")
    all_chunks = build_sparse_embeddings(all_chunks)

    # Embed images
    print(f"Embedding {len(all_pages)} page images...")
    all_pages = embed_page_images(all_pages)

    # Index
    print(f"\nUpdating text index...")
    update_text_index(all_chunks)

    print(f"Updating multimodal index...")
    update_multimodal_index(all_pages)

    # Move to processed/
    for uri in gcs_uris:
        blob_path = uri.replace(f"gs://{BUCKET_NAME}/", "")
        pdf_name = blob_path.split("/")[-1]
        dest_path = f"{PROCESSED_PREFIX}{pdf_name}"
        _move_blob(blob_path, dest_path)
        print(f"Moved {pdf_name} -> {dest_path}")

    summary = {
        "documents_processed": len(gcs_uris),
        "total_chunks": len(all_chunks),
        "total_pages": len(all_pages),
    }
    print(f"\nPipeline complete: {summary}")
    return summary
