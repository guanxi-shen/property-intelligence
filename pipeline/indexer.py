"""Index management for Vertex AI Vector Search.

Writes embedding JSONL and metadata lookup JSONs to GCS, then triggers
index updates via the Vertex AI SDK.
"""

import json
from pathlib import Path
from typing import List, Dict

from google.cloud import storage
from google.cloud import aiplatform

from src.config import (
    PROJECT_ID, CREDENTIALS, BUCKET_NAME, LOCATION,
    TEXT_EMBEDDINGS_PATH, MULTIMODAL_EMBEDDINGS_PATH,
    TEXT_METADATA_PATH, MULTIMODAL_METADATA_PATH,
)


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)


def _upload_json(blob_path: str, data: str):
    """Upload a JSON string to GCS."""
    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_string(data, content_type="application/json")


def _load_existing_metadata(blob_path: str) -> Dict:
    """Load existing metadata lookup from GCS, or return empty dict."""
    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if blob.exists():
        return json.loads(blob.download_as_text())
    return {}


def _load_existing_jsonl(blob_path: str) -> Dict[str, dict]:
    """Load existing JSONL embeddings from GCS, keyed by datapoint id."""
    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return {}
    existing = {}
    for line in blob.download_as_text().strip().split("\n"):
        if line.strip():
            dp = json.loads(line)
            existing[dp["id"]] = dp
    return existing


def _make_text_datapoint_id(chunk: Dict) -> str:
    """Generate a deterministic ID for a text chunk."""
    stem = Path(chunk["source_pdf"]).stem
    # Sanitize: replace spaces and special chars with underscores
    stem = "".join(c if c.isalnum() or c == "_" else "_" for c in stem)
    return f"{stem}_p{chunk['page_number']}_c{chunk['chunk_index']}"


def _make_page_datapoint_id(page: Dict) -> str:
    """Generate a deterministic ID for a page image."""
    stem = Path(page["source_pdf"]).stem
    stem = "".join(c if c.isalnum() or c == "_" else "_" for c in stem)
    return f"{stem}_page_{page['page_number']}"


def _infer_doc_type(filename: str) -> str:
    """Simple heuristic for document type based on filename."""
    lower = filename.lower()
    if "appraisal" in lower:
        return "appraisal"
    if "inspection" in lower:
        return "inspection"
    if "compar" in lower or "comp" in lower:
        return "comparable"
    if "market" in lower:
        return "market_analysis"
    return "property_document"


def _remove_by_source_pdfs(existing: Dict, metadata: Dict, source_pdfs: set, id_fn_prefix: str = ""):
    """Remove all entries from existing JSONL and metadata that belong to the given source PDFs."""
    to_remove = [dp_id for dp_id, meta in metadata.items() if meta.get("source_pdf") in source_pdfs]
    for dp_id in to_remove:
        existing.pop(dp_id, None)
        metadata.pop(dp_id, None)


def update_text_index(chunks: List[Dict]):
    """Write text embeddings as JSONL to GCS and update the text index.

    Merges with existing data. Re-uploaded documents overwrite their old entries.

    Args:
        chunks: List of chunk dicts with embeddings attached.
    """
    existing = _load_existing_jsonl(TEXT_EMBEDDINGS_PATH)
    metadata = _load_existing_metadata(TEXT_METADATA_PATH)

    # Determine which source PDFs are being replaced
    new_source_pdfs = {c["source_pdf"] for c in chunks if c.get("embedding") is not None}
    _remove_by_source_pdfs(existing, metadata, new_source_pdfs)

    for chunk in chunks:
        if chunk.get("embedding") is None:
            continue

        dp_id = _make_text_datapoint_id(chunk)
        doc_type = _infer_doc_type(chunk["source_pdf"])

        datapoint = {
            "id": dp_id,
            "embedding": chunk["embedding"],
            "crowding_tag": chunk["source_pdf"],
            "restricts": [{"namespace": "doc_type", "allow": [doc_type]}],
        }

        if chunk.get("sparse_embedding"):
            datapoint["sparse_embedding"] = chunk["sparse_embedding"]

        existing[dp_id] = datapoint

        metadata[dp_id] = {
            "datapoint_id": dp_id,
            "source_pdf": chunk["source_pdf"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "chunk_type": chunk.get("chunk_type", "text"),
            "content": chunk["text"][:2000],
            "structural_context": chunk.get("structural_context", ""),
            "gcs_uri": f"gs://{BUCKET_NAME}/processed/{chunk['source_pdf']}",
        }

    # Write merged JSONL
    jsonl_content = "\n".join(json.dumps(dp) for dp in existing.values())
    _upload_json(TEXT_EMBEDDINGS_PATH, jsonl_content)
    print(f"Wrote {len(existing)} text datapoints to gs://{BUCKET_NAME}/{TEXT_EMBEDDINGS_PATH}")

    # Write merged metadata
    _upload_json(TEXT_METADATA_PATH, json.dumps(metadata))
    print(f"Updated text metadata: {len(metadata)} total entries")

    _update_index(TEXT_EMBEDDINGS_PATH)


def update_multimodal_index(page_images: List[Dict]):
    """Write page image embeddings as JSONL to GCS and update the multimodal index.

    Merges with existing data. Re-uploaded documents overwrite their old entries.

    Args:
        page_images: List of page dicts with embeddings attached.
    """
    existing = _load_existing_jsonl(MULTIMODAL_EMBEDDINGS_PATH)
    metadata = _load_existing_metadata(MULTIMODAL_METADATA_PATH)

    new_source_pdfs = {p["source_pdf"] for p in page_images if p.get("embedding") is not None}
    _remove_by_source_pdfs(existing, metadata, new_source_pdfs)

    for page in page_images:
        if page.get("embedding") is None:
            continue

        dp_id = _make_page_datapoint_id(page)
        doc_type = _infer_doc_type(page["source_pdf"])

        datapoint = {
            "id": dp_id,
            "embedding": page["embedding"],
            "crowding_tag": page["source_pdf"],
            "restricts": [{"namespace": "doc_type", "allow": [doc_type]}],
        }
        existing[dp_id] = datapoint

        metadata[dp_id] = {
            "datapoint_id": dp_id,
            "source_pdf": page["source_pdf"],
            "page_number": page["page_number"],
            "gcs_uri": page["gcs_uri"],
            "width": page.get("width"),
            "height": page.get("height"),
        }

    # Write merged JSONL
    jsonl_content = "\n".join(json.dumps(dp) for dp in existing.values())
    _upload_json(MULTIMODAL_EMBEDDINGS_PATH, jsonl_content)
    print(f"Wrote {len(existing)} multimodal datapoints to gs://{BUCKET_NAME}/{MULTIMODAL_EMBEDDINGS_PATH}")

    # Write merged metadata
    _upload_json(MULTIMODAL_METADATA_PATH, json.dumps(metadata))
    print(f"Updated multimodal metadata: {len(metadata)} total entries")

    _update_index(MULTIMODAL_EMBEDDINGS_PATH)


def _update_index(embeddings_path: str):
    """Trigger a Vertex AI Vector Search index update from JSONL in GCS.

    contents_delta_uri must be a directory, not a file path.
    """
    aiplatform.init(
        project=PROJECT_ID, location=LOCATION, credentials=CREDENTIALS
    )

    # contents_delta_uri needs the directory containing the JSONL file
    dir_path = "/".join(embeddings_path.split("/")[:-1]) + "/"
    gcs_dir = f"gs://{BUCKET_NAME}/{dir_path}"

    if "text" in embeddings_path:
        index_name = "property-text-index"
    else:
        index_name = "property-multimodal-index"

    indexes = aiplatform.MatchingEngineIndex.list()
    target = None
    for idx in indexes:
        if idx.display_name == index_name:
            target = idx
            break

    if target is None:
        print(f"WARNING: Index '{index_name}' not found. Skipping update.")
        return

    target.update_embeddings(
        contents_delta_uri=gcs_dir,
        is_complete_overwrite=True,
    )
    print(f"Triggered index update for '{index_name}' from {gcs_dir}")
