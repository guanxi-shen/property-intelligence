"""FastAPI server with SSE streaming chat and pipeline trigger endpoints"""

import asyncio
import json
import logging
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from google.cloud import storage

from src.config import CREDENTIALS, PROJECT_ID, BUCKET_NAME, PROCESSED_PREFIX

logger = logging.getLogger(__name__)
app = FastAPI(title="Property Intelligence")

_agents = {}
_storage_client = None


def _get_storage():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)
    return _storage_client


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
