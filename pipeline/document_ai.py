"""Document AI Layout Parser integration for PDF parsing.

Uses the pretrained Layout Parser v1.6 (Gemini 3.0 Flash) to extract
semantic chunks from property PDFs while preserving document structure.

Large PDFs are split into <=30 page segments with PyMuPDF and sent
as inline content to the online API (no batch API needed).
"""

import os
import tempfile
from typing import List, Dict

import fitz  # PyMuPDF
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions

from src.config import (
    PROJECT_ID, PROJECT_NUMBER, DOCUMENT_AI_PROCESSOR_ID, DOCUMENT_AI_LOCATION,
    CREDENTIALS, BUCKET_NAME,
)

PROCESSOR_VERSION = "pretrained-layout-parser-v1.6-2026-01-13"
ONLINE_PAGE_LIMIT = 15


def _get_processor_name() -> str:
    project = PROJECT_NUMBER or PROJECT_ID
    return (
        f"projects/{project}/locations/{DOCUMENT_AI_LOCATION}"
        f"/processors/{DOCUMENT_AI_PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )


def _get_docai_client() -> documentai.DocumentProcessorServiceClient:
    opts = ClientOptions(
        api_endpoint=f"{DOCUMENT_AI_LOCATION}-documentai.googleapis.com"
    )
    return documentai.DocumentProcessorServiceClient(
        client_options=opts, credentials=CREDENTIALS
    )


def _classify_chunk_type(content: str) -> str:
    """Heuristic chunk type from content text."""
    lower = content.lower()
    if any(kw in lower for kw in ['table', '|', '\t']):
        return "table"
    return "text"


def _parse_document_chunks(document: documentai.Document, source_pdf: str, page_offset: int = 0) -> List[Dict]:
    """Convert a Document AI Document into a list of chunk dicts.

    page_offset adjusts page numbers when processing split segments.
    """
    chunks = []

    if document.chunked_document and document.chunked_document.chunks:
        for idx, chunk in enumerate(document.chunked_document.chunks):
            content = chunk.content.strip() if chunk.content else ""
            if not content:
                continue

            page_num = (chunk.page_span.page_start + 1 + page_offset) if chunk.page_span else (1 + page_offset)

            heading = ""
            if chunk.page_headers:
                heading = " > ".join(
                    h.text.strip() for h in chunk.page_headers if h.text
                )

            chunks.append({
                "text": content,
                "source_pdf": source_pdf,
                "page_number": page_num,
                "chunk_type": _classify_chunk_type(content),
                "chunk_index": idx,
                "structural_context": heading,
            })
    elif document.text:
        for page in document.pages:
            page_text = _extract_page_text(document.text, page)
            if page_text.strip():
                chunks.append({
                    "text": page_text.strip(),
                    "source_pdf": source_pdf,
                    "page_number": page.page_number + page_offset,
                    "chunk_type": "text",
                    "chunk_index": len(chunks),
                    "structural_context": "",
                })

    return chunks


def _extract_page_text(full_text: str, page) -> str:
    """Extract text for a specific page using layout text anchors."""
    segments = []
    if page.layout and page.layout.text_anchor and page.layout.text_anchor.text_segments:
        for seg in page.layout.text_anchor.text_segments:
            start = int(seg.start_index) if seg.start_index else 0
            end = int(seg.end_index) if seg.end_index else 0
            segments.append(full_text[start:end])
    return "".join(segments)


def _process_inline(pdf_bytes: bytes, source_pdf: str, page_offset: int = 0) -> List[Dict]:
    """Process PDF bytes via online API with inline content."""
    client = _get_docai_client()
    processor_name = _get_processor_name()

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=pdf_bytes, mime_type="application/pdf"
        ),
        process_options=documentai.ProcessOptions(
            layout_config=documentai.ProcessOptions.LayoutConfig(
                chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                    chunk_size=500,
                    include_ancestor_headings=True,
                )
            )
        ),
    )

    result = client.process_document(request=request, timeout=600)
    return _parse_document_chunks(result.document, source_pdf, page_offset)


def parse_pdf(gcs_uri: str) -> List[Dict]:
    """Parse a PDF by downloading locally and processing via online API.

    For PDFs >30 pages, splits into segments with PyMuPDF and processes
    each segment separately. All processing uses the online API.
    """
    source_pdf = gcs_uri.rstrip("/").split("/")[-1]

    # Download PDF from GCS
    storage_client = storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)
    path = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")
    blob = storage_client.bucket(BUCKET_NAME).blob(path)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        blob.download_to_filename(tmp_path)

    try:
        doc = fitz.open(tmp_path)
        total_pages = len(doc)

        if total_pages <= ONLINE_PAGE_LIMIT:
            pdf_bytes = doc.tobytes()
            doc.close()
            print(f"  Online API: {total_pages} pages")
            return _process_inline(pdf_bytes, source_pdf)

        # Large PDF: split into segments
        print(f"  Splitting {total_pages} pages into {ONLINE_PAGE_LIMIT}-page segments")
        segments = []
        for start in range(0, total_pages, ONLINE_PAGE_LIMIT):
            end = min(start + ONLINE_PAGE_LIMIT, total_pages)
            segment_doc = fitz.open()
            segment_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            segments.append((start, end, segment_doc.tobytes()))
            segment_doc.close()

        doc.close()

        all_chunks = []
        for start, end, segment_bytes in segments:
            print(f"  Processing pages {start + 1}-{end}...")
            chunks = _process_inline(segment_bytes, source_pdf, page_offset=start)
            all_chunks.extend(chunks)

        for i, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = i

        return all_chunks

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
