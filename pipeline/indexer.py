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


def update_text_index(chunks: List[Dict]):
    """Write text embeddings as JSONL to GCS and update the text index.

    Each chunk dict must have: text, source_pdf, page_number, chunk_index,
    chunk_type, embedding, sparse_embedding.

    Args:
        chunks: List of chunk dicts with embeddings attached.
    """
    # Build JSONL and metadata
    jsonl_lines = []
    metadata = _load_existing_metadata(TEXT_METADATA_PATH)

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

        # Attach sparse embedding for hybrid search
        if chunk.get("sparse_embedding"):
            datapoint["sparse_embedding"] = chunk["sparse_embedding"]

        jsonl_lines.append(json.dumps(datapoint))

        metadata[dp_id] = {
            "datapoint_id": dp_id,
            "source_pdf": chunk["source_pdf"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "chunk_type": chunk.get("chunk_type", "text"),
            "content": chunk["text"][:2000],  # Truncate for metadata lookup
            "structural_context": chunk.get("structural_context", ""),
            "gcs_uri": f"gs://{BUCKET_NAME}/processed/{chunk['source_pdf']}",
        }

    # Upload JSONL
    jsonl_content = "\n".join(jsonl_lines)
    _upload_json(TEXT_EMBEDDINGS_PATH, jsonl_content)
    print(f"Wrote {len(jsonl_lines)} text datapoints to gs://{BUCKET_NAME}/{TEXT_EMBEDDINGS_PATH}")

    # Upload metadata
    _upload_json(TEXT_METADATA_PATH, json.dumps(metadata))
    print(f"Updated text metadata: {len(metadata)} total entries")

    # Trigger index update
    _update_index(TEXT_EMBEDDINGS_PATH)


def update_multimodal_index(page_images: List[Dict]):
    """Write page image embeddings as JSONL to GCS and update the multimodal index.

    Each page dict must have: source_pdf, page_number, gcs_uri, embedding.

    Args:
        page_images: List of page dicts with embeddings attached.
    """
    jsonl_lines = []
    metadata = _load_existing_metadata(MULTIMODAL_METADATA_PATH)

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
        jsonl_lines.append(json.dumps(datapoint))

        metadata[dp_id] = {
            "datapoint_id": dp_id,
            "source_pdf": page["source_pdf"],
            "page_number": page["page_number"],
            "gcs_uri": page["gcs_uri"],
            "width": page.get("width"),
            "height": page.get("height"),
        }

    # Upload JSONL
    jsonl_content = "\n".join(jsonl_lines)
    _upload_json(MULTIMODAL_EMBEDDINGS_PATH, jsonl_content)
    print(f"Wrote {len(jsonl_lines)} multimodal datapoints to gs://{BUCKET_NAME}/{MULTIMODAL_EMBEDDINGS_PATH}")

    # Upload metadata
    _upload_json(MULTIMODAL_METADATA_PATH, json.dumps(metadata))
    print(f"Updated multimodal metadata: {len(metadata)} total entries")

    # Trigger index update
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
