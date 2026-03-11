"""FastAPI server with SSE streaming chat, file upload with pipeline, and trigger endpoints"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from google.cloud import storage

from src.config import CREDENTIALS, PROJECT_ID, BUCKET_NAME, PROCESSED_PREFIX, UPLOADS_PREFIX

logger = logging.getLogger(__name__)
app = FastAPI(title="Property Intelligence")

_agents = {}
_storage_client = None

# Upload constraints
UPLOAD_MAX_FILES = 10
UPLOAD_MAX_SIZE_MB = 50
UPLOAD_DAILY_LIMIT = 30
_upload_counts: dict[str, int] = defaultdict(int)


def _get_storage():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)
    return _storage_client


def _check_daily_limit(count: int) -> str | None:
    """Return error message if daily limit would be exceeded, else None."""
    today = date.today().isoformat()
    if _upload_counts[today] + count > UPLOAD_DAILY_LIMIT:
        remaining = UPLOAD_DAILY_LIMIT - _upload_counts[today]
        return f"Daily upload limit ({UPLOAD_DAILY_LIMIT}). {remaining} remaining today."
    return None


def _validate_pdf(pdf_bytes: bytes, filename: str) -> str | None:
    """Return error message if PDF is invalid, else None."""
    import fitz
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > UPLOAD_MAX_SIZE_MB:
        return f"{filename}: {size_mb:.1f}MB exceeds {UPLOAD_MAX_SIZE_MB}MB limit"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = doc.page_count
        doc.close()
    except Exception:
        return f"{filename}: not a valid PDF"
    if pages == 0:
        return f"{filename}: PDF has no pages"
    return None


def _get_agent(session_id: str):
    from src.agent import PropertyAgent
    if session_id not in _agents:
        _agents[session_id] = PropertyAgent()
    return _agents[session_id]


def _enrich_citations(citations: list) -> list:
    """Add pdf_url (signed URL to full PDF#page=N) to every citation."""
    client = _get_storage()
    bucket = client.bucket(BUCKET_NAME)
    pdf_url_cache = {}

    for c in citations:
        pdf_name = c.get("source_pdf", "")
        page = c.get("page_number", 1)

        if pdf_name not in pdf_url_cache:
            for prefix in [PROCESSED_PREFIX, "uploads/"]:
                blob = bucket.blob(f"{prefix}{pdf_name}")
                if blob.exists():
                    try:
                        pdf_url_cache[pdf_name] = blob.generate_signed_url(
                            version="v4",
                            expiration=timedelta(days=7),
                            method="GET",
                        )
                    except Exception as e:
                        logger.warning(f"Signed URL error for {pdf_name}: {e}")
                        pdf_url_cache[pdf_name] = ""
                    break
            else:
                pdf_url_cache[pdf_name] = ""

        base_url = pdf_url_cache[pdf_name]
        c["pdf_url"] = f"{base_url}#page={page}" if base_url else ""

    return citations


# -- Streaming chat endpoint (SSE) --

def _sse_event(event_type: str, data) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE streaming chat. Streams thinking, tool calls, and answer text live."""
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    agent = _get_agent(session_id)
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def stream_cb(event_type, data):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    async def run_agent():
        try:
            result = await asyncio.to_thread(agent.chat, message, stream_cb)
            result["citations"] = _enrich_citations(result.get("citations", []))
            result["retrieved_docs"] = _enrich_citations(result.get("retrieved_docs", []))
            await queue.put(("done", result))
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            await queue.put(("done", {
                "answer": f"Error: {e}", "citations": [],
                "retrieved_docs": [], "thinking": "",
            }))

    async def event_generator():
        task = asyncio.create_task(run_agent())
        while True:
            event_type, data = await queue.get()
            if event_type == "done":
                yield _sse_event("done", data)
                break
            yield _sse_event(event_type, data)
        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -- Non-streaming fallback --

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    agent = _get_agent(session_id)
    result = agent.chat(message)
    result["citations"] = _enrich_citations(result.get("citations", []))
    result["retrieved_docs"] = _enrich_citations(result.get("retrieved_docs", []))
    return result


# -- Upload + Process (single action with SSE progress) --

@app.post("/upload")
async def upload_and_process(files: List[UploadFile] = File(...)):
    """Validate, upload to GCS, and run full pipeline. Returns SSE progress stream."""
    if len(files) > UPLOAD_MAX_FILES:
        return JSONResponse(
            {"error": f"Max {UPLOAD_MAX_FILES} files per upload, got {len(files)}"},
            status_code=400,
        )

    limit_err = _check_daily_limit(len(files))
    if limit_err:
        return JSONResponse({"error": limit_err}, status_code=429)

    # Read all file bytes upfront and validate
    file_data = []  # [(filename, bytes)]
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            return JSONResponse({"error": f"{f.filename}: only PDF files allowed"}, status_code=400)
        raw = await f.read()
        err = _validate_pdf(raw, f.filename)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        file_data.append((f.filename, raw))

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(step: str, status: str, detail: str = ""):
        loop.call_soon_threadsafe(queue.put_nowait, {
            "step": step, "status": status, "detail": detail,
        })

    def run_pipeline():
        from concurrent.futures import ThreadPoolExecutor
        from pipeline.document_ai import parse_pdf
        from pipeline.page_converter import convert_pdf_to_pages
        from pipeline.embedder import embed_text_chunks, embed_page_images, build_sparse_embeddings
        from pipeline.indexer import update_text_index, update_multimodal_index

        n = len(file_data)
        client = _get_storage()
        bucket = client.bucket(BUCKET_NAME)

        # Upload to GCS (shared prerequisite)
        emit("upload", "active", f"Uploading {n} PDF(s) to Cloud Storage")
        for i, (name, raw) in enumerate(file_data):
            blob = bucket.blob(f"{UPLOADS_PREFIX}{name}")
            blob.upload_from_string(raw, content_type="application/pdf")
            emit("upload", "active", f"Uploaded {i+1}/{n}: {name}")
        emit("upload", "done", f"{n} file(s) uploaded")

        gcs_uris = [(name, f"gs://{BUCKET_NAME}/{UPLOADS_PREFIX}{name}") for name, _ in file_data]

        # Two fully independent branches after upload
        def text_branch():
            """Parse → Embed text + TF-IDF → Update text index"""
            all_chunks = []

            emit("parse", "active", f"Parsing {n} document(s) with Layout Parser")
            for i, (name, uri) in enumerate(gcs_uris):
                try:
                    chunks = parse_pdf(uri)
                    for c in chunks:
                        c["source_pdf"] = name
                    all_chunks.extend(chunks)
                    emit("parse", "active", f"Parsed {i+1}/{n}: {name} ({len(chunks)} chunks)")
                except Exception as e:
                    emit("parse", "active", f"Parse error on {name}: {e}")
            emit("parse", "done", f"{len(all_chunks)} chunks extracted")

            # Dense and sparse embeddings run in parallel (both read "text", write to different keys)
            def _dense():
                emit("embed_text", "active", f"Embedding {len(all_chunks)} text chunks")
                embed_text_chunks(all_chunks)
                emit("embed_text", "done", f"{len(all_chunks)} text embeddings generated")

            def _sparse():
                emit("sparse", "active", "Building TF-IDF sparse vectors")
                build_sparse_embeddings(all_chunks)
                emit("sparse", "done", "Sparse vectors built")

            with ThreadPoolExecutor(max_workers=2) as inner_pool:
                d = inner_pool.submit(_dense)
                s = inner_pool.submit(_sparse)
                d.result()
                s.result()

            emit("text_index", "active", "Updating text search index")
            update_text_index(all_chunks)
            emit("text_index", "done", "Text index updated")

        def image_branch():
            """Render pages → Embed images → Update multimodal index"""
            all_pages = []

            emit("render", "active", f"Rendering pages to PNG")
            for i, (name, uri) in enumerate(gcs_uris):
                try:
                    pages = convert_pdf_to_pages(uri)
                    all_pages.extend(pages)
                    emit("render", "active", f"Rendered {i+1}/{n}: {name} ({len(pages)} pages)")
                except Exception as e:
                    emit("render", "active", f"Render error on {name}: {e}")
            emit("render", "done", f"{len(all_pages)} page images created")

            emit("embed_images", "active", f"Embedding {len(all_pages)} page images")
            all_pages = embed_page_images(all_pages)
            embedded = sum(1 for p in all_pages if p.get("embedding"))
            emit("embed_images", "done", f"{embedded} image embeddings generated")

            emit("mm_index", "active", "Updating multimodal search index")
            update_multimodal_index(all_pages)
            emit("mm_index", "done", "Multimodal index updated")

        with ThreadPoolExecutor(max_workers=2) as pool:
            text_future = pool.submit(text_branch)
            image_future = pool.submit(image_branch)
            text_future.result()
            image_future.result()

        # Finalize: move to processed/
        emit("finalize", "active", "Moving files to processed/")
        for name, _ in file_data:
            try:
                src = bucket.blob(f"{UPLOADS_PREFIX}{name}")
                bucket.copy_blob(src, bucket, f"{PROCESSED_PREFIX}{name}")
                src.delete()
            except Exception:
                pass

        _upload_counts[date.today().isoformat()] += n
        emit("finalize", "done", f"{n} document(s) ready to query")

    async def event_generator():
        task = asyncio.get_event_loop().run_in_executor(None, run_pipeline)

        # Emit events until pipeline finishes
        done = False
        while not done:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield _sse_event("progress", event)
                if event["step"] == "finalize" and event["status"] == "done":
                    done = True
            except asyncio.TimeoutError:
                # Check if pipeline thread crashed
                if task.done():
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        yield _sse_event("progress", {
                            "step": "error", "status": "error",
                            "detail": str(exc),
                        })
                    done = True

        yield _sse_event("complete", {"message": "Pipeline complete"})
        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/upload/status")
async def upload_status():
    """Return daily upload count and limits."""
    today = date.today().isoformat()
    used = _upload_counts[today]
    return {
        "daily_limit": UPLOAD_DAILY_LIMIT,
        "used_today": used,
        "remaining": UPLOAD_DAILY_LIMIT - used,
        "max_files_per_upload": UPLOAD_MAX_FILES,
        "max_size_mb": UPLOAD_MAX_SIZE_MB,
    }


# -- Pipeline triggers --

@app.post("/pipeline/trigger")
async def pipeline_trigger(request: Request):
    body = await request.json()
    bucket = body.get("bucket", "")
    name = body.get("name", "")

    if not name.startswith("uploads/") or not name.lower().endswith(".pdf"):
        return {"status": "skipped", "reason": f"Not a PDF in uploads/: {name}"}

    gcs_uri = f"gs://{bucket}/{name}"
    logger.info(f"Processing triggered for: {gcs_uri}")

    try:
        from pipeline.processor import process_document
        result = process_document(gcs_uri)
        return {"status": "processed", "gcs_uri": gcs_uri, "result": result}
    except Exception as e:
        logger.error(f"Pipeline error for {gcs_uri}: {e}", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.post("/pipeline/run")
async def pipeline_run_all():
    try:
        from pipeline.processor import process_all_new
        result = process_all_new()
        return {"status": "complete", "result": result}
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/")
async def root():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health():
    return {"status": "ok"}
