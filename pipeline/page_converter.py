"""PDF page-to-PNG converter using PyMuPDF.

Downloads PDFs from GCS, renders each page at 200 DPI, and uploads
the resulting PNGs back to GCS for multimodal embedding and agent viewing.
"""

import tempfile
import os
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
from google.cloud import storage

from src.config import BUCKET_NAME, CREDENTIALS, PROJECT_ID, PAGE_IMAGES_PREFIX

DPI = 200
ZOOM = DPI / 72  # PyMuPDF default is 72 DPI


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)


def _pdf_stem(gcs_uri: str) -> str:
    """Extract filename stem from GCS URI (no extension)."""
    filename = gcs_uri.rstrip("/").split("/")[-1]
    return Path(filename).stem


def convert_pdf_to_pages(gcs_uri: str) -> List[Dict]:
    """Render each page of a PDF to PNG and upload to GCS.

    Args:
        gcs_uri: GCS URI of the source PDF.

    Returns:
        List of dicts, each with: page_number, gcs_uri, width, height
    """
    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob_path = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")
    stem = _pdf_stem(gcs_uri)

    # Download PDF to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        bucket.blob(blob_path).download_to_filename(tmp_path)

    try:
        doc = fitz.open(tmp_path)
        page_images = []
        mat = fitz.Matrix(ZOOM, ZOOM)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)

            # Render to PNG bytes
            png_bytes = pix.tobytes("png")

            # Upload to GCS
            dest = f"{PAGE_IMAGES_PREFIX}{stem}/page_{page_num + 1}.png"
            dest_blob = bucket.blob(dest)
            dest_blob.upload_from_string(png_bytes, content_type="image/png")

            page_images.append({
                "page_number": page_num + 1,
                "gcs_uri": f"gs://{BUCKET_NAME}/{dest}",
                "width": pix.width,
                "height": pix.height,
                "source_pdf": gcs_uri.rstrip("/").split("/")[-1],
            })

        doc.close()
        print(f"Rendered {len(page_images)} pages from {stem}")
        return page_images

    finally:
        os.unlink(tmp_path)
