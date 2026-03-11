<p align="center">
  <h1 align="center">Property Intelligence</h1>
  <p align="center">
    Multimodal RAG system for property document analysis<br/>
    powered by Gemini, Document AI, and Vertex AI Vector Search
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=googlecloud&logoColor=white" alt="Google Cloud"/>
  <img src="https://img.shields.io/badge/Gemini-886FBF?style=flat&logo=googlegemini&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/Cloud_Run-4285F4?style=flat&logo=googlecloud&logoColor=white" alt="Cloud Run"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
</p>

<p align="center">
  <a href="https://property-intelligence-963905106335.us-central1.run.app"><strong>Live Demo</strong></a>
</p>

---

## Problem

A large real estate customer needs a property valuation chatbot, but their proprietary data is trapped in siloed, unstructured PDF formats that standard RAG tools can't parse. Property documents -- appraisals, inspection reports, comparable sales grids, plat maps, settlement statements -- contain a mix of dense text, structured tables, photographs, hand-drawn sketches, and scanned forms. Standard text-extraction pipelines lose table structure, miss visual content entirely, and can't distinguish a comparable sales adjustment grid from a paragraph of boilerplate.

## How This Solves It

**1. Structure-aware parsing instead of blind text extraction**

Document AI Layout Parser (powered by Gemini 3.0 Flash) understands document structure -- headings, tables, lists, paragraphs -- and produces semantic chunks that preserve context. A row in an adjustment grid stays with its column headers. A property address stays associated with its valuation figures. This is fundamentally different from naive page-level or sliding-window chunking that breaks mid-table.

**2. Dual-modality retrieval -- text and visual**

Many answers in property documents live in visual elements: adjustment grids, floor plans, neighborhood maps, property photos, site sketches. Text extraction alone cannot capture these. The system embeds both extracted text chunks and rendered page images into the same 768-dimensional semantic space using Gemini Embedding 2, then indexes them in two separate Vertex AI Vector Search indexes. The agent decides at query time whether to search text, images, or both.

**3. Hybrid search for exact and semantic matching**

Property queries often involve exact values -- "$425,000", "123 Main St", "MLS #2024-1234". Pure dense vector search struggles with exact keyword matches. The text index combines dense embeddings with TF-IDF sparse vectors, fused via Reciprocal Rank Fusion (RRF). This means semantic queries ("properties with deferred maintenance") and exact keyword queries ("$425,000 adjusted sale price") both work reliably.

**4. Agentic reasoning with self-retry and discovery**

Rather than dumping all retrieved context into a single prompt, a Gemini 3.1 Pro agent with function calling autonomously plans its search strategy. It formulates diverse queries (mixing semantic and keyword phrasings), evaluates the results, and if they're insufficient, refines its queries and searches again -- up to 3 rounds per tool. Cross-round deduplication ensures repeated searches surface new information rather than redundant results. The agent can call `search_text` for factual lookups, `search_pages` for visual analysis, or both in sequence, deciding on the fly based on what it finds. Every claim in the response is inline-cited to a specific PDF page.

**5. Multimodal document analysis**

When the agent retrieves page images via `search_pages`, Gemini doesn't just return metadata -- it visually analyzes the actual page renders. Adjustment grids, floor plans, property photos, and scanned forms are passed as image content directly into the model's context via `FunctionResponseFileData`. This means the agent can read values from a comparable sales grid, describe what it sees in a property photo, or trace lines on a plat map -- answering questions that pure text retrieval would miss entirely.

**6. Self-service document ingestion**

The system includes a browser-based upload flow where users drop PDFs and watch the full ingestion pipeline execute in real time -- parse, render, embed (text and images in parallel), index. New documents are immediately searchable. Re-uploading the same document overwrites its previous entries. This turns a batch ETL process into something a non-technical user can operate.

## Architecture

```
                    User Browser
                         |
                         | SSE streaming
                         v
+----------------------------------------------------------------+
|  Cloud Run (FastAPI)                                           |
|                                                                |
|  POST /chat/stream                                             |
|       |                                                        |
|       v                                                        |
|  Gemini 3.1 Pro Agent (function calling, multi-round)          |
|       |                    |                                   |
|       | search_text        | search_pages                     |
|       v                    v                                   |
|  +-----------+   +----------------+                            |
|  | Text      |   | Multimodal     |                            |
|  | Index     |   | Index          |    Vertex AI Vector Search |
|  | dense +   |   | dense          |                            |
|  | sparse    |   |                |                            |
|  +-----------+   +----------------+                            |
|       ^                    ^                                   |
|       |                    |                                   |
|  POST /upload              |                                   |
|       |                    |                                   |
|       v                    |                                   |
|  Cloud Storage (uploads/)  |                                   |
|       |                    |                                   |
|       +----------+---------+                                   |
|       |  (parallel branches)                                   |
|       |                    |                                   |
|  TEXT PIPELINE        IMAGE PIPELINE                           |
|       |                    |                                   |
|  Document AI          PyMuPDF                                  |
|  Layout Parser        page render                              |
|       |                    |                                   |
|   +---+---+           Gemini Emb 2                             |
|   |       |           (dense 768d)                             |
| Gemini  TF-IDF             |                                   |
| Emb 2   (sparse)           |                                   |
|   |       |                |                                   |
|   +---+---+                |                                   |
|       |                    |                                   |
|       v                    v                                   |
|  Update text index    Update multimodal index                  |
|       |                    |                                   |
|       +----------+---------+                                   |
|                  |                                             |
|             Finalize                                           |
+----------------------------------------------------------------+
```

### Query Flow

```
  User Question
       |
       v
  Gemini 3.1 Pro (thinking + function calling)
       |
       +--- Round 1: search_text(["property value 123 Main St", "$425,000"])
       |       |
       |       v
       |    Text Index ---> dense + sparse (RRF) ---> ranked chunks
       |       |
       |       v
       |    Agent evaluates results ---> insufficient, refines query
       |
       +--- Round 2: search_pages(["adjustment grid", "comparable sales table"])
       |       |
       |       v
       |    Multimodal Index ---> image similarity ---> page PNGs
       |       |
       |       v
       |    Gemini visually analyzes page images
       |    (reads tables, describes photos, traces plat maps)
       |       |
       |       v
       |    Agent evaluates results ---> sufficient
       |
       v
  Cited Answer (SSE streamed with inline [source.pdf, p.N] citations)
```

### Ingestion Pipeline

Two fully independent pipelines run in parallel after upload:

```
1. Upload to GCS
         |
         +------- TEXT PIPELINE -------+------- IMAGE PIPELINE ------+
         |                             |                              |
    2. Document AI Layout Parser       2. Render pages to PNG         |
         |                             |                              |
      +--+--+                          3. Image embeddings (768d)     |
      |     |                          |                              |
  3a. Dense  3b. TF-IDF               4. Update multimodal index     |
  embeddings  sparse vectors           |                              |
      |     |                          +------------------------------+
      +--+--+                          |
         |                             |
    4. Update text index               |
         |                             |
         +-----------------------------+
         |
    Ready to query
```

| Stage | Service | What it does |
|-------|---------|-------------|
| **Parse** | Document AI Layout Parser | Extracts semantic text chunks with structural context (headings, tables) |
| **Render** | PyMuPDF | Renders each page to 200 DPI PNG, uploads to GCS |
| **Text Embed** (3a) | Gemini Embedding 2 | 768-dim dense vectors for text chunks (`RETRIEVAL_DOCUMENT` task type) |
| **Sparse Embed** (3b) | scikit-learn TF-IDF | Sparse vectors for exact keyword matching (addresses, dollar amounts) -- runs parallel with dense |
| **Image Embed** | Gemini Embedding 2 | 768-dim dense vectors for page images (shared semantic space with text) |
| **Index** | Vertex AI Vector Search | Upserts vectors with crowding tags and doc_type restricts; merges with existing data, same-filename documents are overwritten |

### Search

The Gemini agent has two search tools and decides which to call (can use both, multiple rounds):

| Tool | Index | Method | Best for |
|------|-------|--------|----------|
| `search_text` | Text (hybrid) | Dense + TF-IDF sparse via RRF fusion (alpha=0.85) | Facts, numbers, addresses, dollar amounts |
| `search_pages` | Multimodal (dense) | Image embedding similarity with diversity reranking | Tables, photos, maps, floor plans, sketches |

### Frontend

- **Chat**: SSE streaming of thinking, tool calls, and answer text
- **Citations**: Clickable inline citations open a PDF viewer at the cited page
- **Upload**: Drag-and-drop file picker with real-time pipeline progress timeline showing parallel execution
- **Retrieved docs**: Collapsible panel showing all retrieved documents grouped by source
- Google Material Design styling with Google Sans typography

## Project Structure

```
property-intelligence/
|-- api/
|   `-- server.py              # FastAPI with SSE streaming + pipeline trigger
|-- frontend/
|   `-- index.html             # Single-page app with PDF viewer
|-- pipeline/
|   |-- processor.py           # Orchestrator: parse -> render -> embed -> index
|   |-- document_ai.py         # Document AI Layout Parser integration
|   |-- page_converter.py      # PDF page to PNG rendering
|   |-- embedder.py            # Dense + sparse embedding generation
|   `-- indexer.py             # Vertex AI Vector Search upsert
|-- src/
|   |-- agent.py               # Gemini agent with function calling + citation parsing
|   |-- llm.py                 # LLM wrapper with streaming and 429 retry
|   |-- text_search.py         # Hybrid text search (dense + TF-IDF)
|   |-- visual_search.py       # Multimodal page image search
|   |-- prompts.py             # System prompt and tool declarations
|   |-- config.py              # Environment-based configuration
|   `-- utils.py               # GCS signed URL generation
|-- scripts/
|   |-- create_indexes.py      # One-time: provision Vector Search indexes + endpoints
|   `-- seed_data.py           # One-time: batch process initial PDFs
|-- .github/workflows/
|   `-- deploy.yml             # CI/CD: deploy to Cloud Run + Eventarc trigger
|-- Dockerfile
|-- requirements.txt
`-- .env                       # Local secrets (gitignored)
```

## Getting Started

### Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI installed and authenticated
- Python 3.11+
- Enable these APIs in your project:
  ```bash
  gcloud services enable \
    storage.googleapis.com \
    documentai.googleapis.com \
    aiplatform.googleapis.com \
    run.googleapis.com \
    eventarc.googleapis.com
  ```

### Step 1: Clone and install

```bash
git clone https://github.com/guanxi-shen/property-intelligence.git
cd property-intelligence
pip install -r requirements.txt
```

### Step 2: Create a service account

```bash
gcloud iam service-accounts create property-intelligence \
  --display-name="Property Intelligence"

# Grant required roles
SA_EMAIL="property-intelligence@YOUR_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/documentai.editor"

# Download key
gcloud iam service-accounts keys create sa-key.json \
  --iam-account=$SA_EMAIL
```

### Step 3: Create GCS bucket

```bash
gsutil mb -l us-central1 gs://your-bucket-name
```

### Step 4: Create Document AI processor

Go to **Cloud Console > Document AI > Create Processor**, select **Layout Parser**, and note the processor ID.

### Step 5: Create `.env`

```env
GCP_PROJECT_ID=your-project-id
GCP_PROJECT_NUMBER=123456789
GOOGLE_CLOUD_REGION=us-central1

BUCKET_NAME=your-bucket-name

# Leave blank for now -- Step 6 will populate these
TEXT_SEARCH_API_ENDPOINT=
TEXT_SEARCH_INDEX_ENDPOINT=
TEXT_SEARCH_DEPLOYED_INDEX_ID=

MULTIMODAL_SEARCH_API_ENDPOINT=
MULTIMODAL_SEARCH_INDEX_ENDPOINT=
MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID=

DOCUMENT_AI_PROCESSOR_ID=your-processor-id
DOCUMENT_AI_LOCATION=us

credentials_dict=<paste contents of sa-key.json as a single line>
```

### Step 6: Create Vector Search indexes

```bash
python scripts/create_indexes.py
```

This script:
1. Uploads seed embeddings to GCS (required for index creation)
2. Creates two Vertex AI Vector Search indexes:
   - **property-text-index** -- hybrid dense + sparse for text chunks
   - **property-multimodal-index** -- dense only for page images
3. Creates public endpoints and deploys indexes to them
4. Prints the endpoint values to copy into `.env`

```
============================================================
Add these to your .env file:
============================================================
TEXT_SEARCH_INDEX_ENDPOINT=projects/123456/locations/us-central1/indexEndpoints/...
TEXT_SEARCH_API_ENDPOINT=1234567890.us-central1-123456.vdb.vertexai.goog
TEXT_SEARCH_DEPLOYED_INDEX_ID=property_text_deployed

MULTIMODAL_SEARCH_INDEX_ENDPOINT=projects/123456/locations/us-central1/indexEndpoints/...
MULTIMODAL_SEARCH_API_ENDPOINT=9876543210.us-central1-123456.vdb.vertexai.goog
MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID=property_multimodal_deployed
============================================================
```

Copy these values into your `.env`. Index deployment takes ~30 minutes.

### Step 7: Upload documents and run pipeline

```bash
# Upload PDFs to GCS
gsutil cp your_documents/*.pdf gs://your-bucket-name/uploads/

# Process all PDFs (parse, render, embed, index)
python scripts/seed_data.py
```

### Step 8: Run locally

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 in your browser.

## Deploy to Cloud Run

Deployment is automated via GitHub Actions on push to `main`.

### GitHub Secrets

Add these in **repo Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_PROJECT_NUMBER` | Google Cloud project number |
| `CREDENTIALS_DICT` | Service account JSON (single-line) |
| `BUCKET_NAME` | GCS bucket name |
| `TEXT_SEARCH_API_ENDPOINT` | From `create_indexes.py` output |
| `TEXT_SEARCH_INDEX_ENDPOINT` | From `create_indexes.py` output |
| `TEXT_SEARCH_DEPLOYED_INDEX_ID` | From `create_indexes.py` output |
| `MULTIMODAL_SEARCH_API_ENDPOINT` | From `create_indexes.py` output |
| `MULTIMODAL_SEARCH_INDEX_ENDPOINT` | From `create_indexes.py` output |
| `MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID` | From `create_indexes.py` output |
| `DOCUMENT_AI_PROCESSOR_ID` | Document AI processor ID |
| `SERVICE_ACCOUNT_EMAIL` | Service account email (for Eventarc trigger) |

Push to `main` to trigger deployment. The workflow:
1. Builds the container with Cloud Build
2. Deploys to Cloud Run
3. Creates an Eventarc trigger so new PDFs in GCS are processed automatically

### Auto-processing

Once deployed, uploading a PDF to `gs://your-bucket/uploads/` triggers the full pipeline automatically via Eventarc. Monitor with:

```bash
gcloud run logs tail property-intelligence --region=us-central1
```

## GCP Services

| Service | Purpose |
|---------|---------|
| **Cloud Storage** | PDF uploads, page images, embeddings, metadata |
| **Document AI** | Layout Parser for semantic PDF chunking |
| **Vertex AI Vector Search** | Two managed indexes (text hybrid + multimodal) |
| **Gemini 3.1 Pro** | Agentic reasoning with function calling |
| **Gemini Embedding 2** | Text and image embeddings (shared 768-dim space) |
| **Cloud Run** | Hosts the FastAPI application |
| **Eventarc** | Routes GCS upload events to Cloud Run |

## License

[MIT](LICENSE)
