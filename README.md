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

---

Upload property documents (appraisals, inspections, comparable sales) and ask questions in natural language. The system parses PDFs with Document AI Layout Parser, embeds text and page images into Vertex AI Vector Search, and answers queries with Gemini using a multimodal agentic RAG pipeline with inline citations and live streaming.

## Architecture

```
                         +------------------+
                         |   GCS Bucket     |
                         |  uploads/*.pdf   |
                         +--------+---------+
                                  |
                         Eventarc trigger
                                  |
                                  v
+-------------+    SSE    +-------+--------+    gRPC    +--------------------+
|   Browser   | <-------> |   Cloud Run    | <--------> | Vertex AI Vector   |
|  (frontend) |           |   (FastAPI)    |            | Search (2 indexes) |
+-------------+           +---+----+---+---+            +--------------------+
                              |    |   |
                   +----------+    |   +----------+
                   v               v              v
              Document AI     Gemini 3.1      Gemini Embedding
             Layout Parser       Pro              2
```

### Pipeline (per document)

1. **Parse** -- Document AI Layout Parser extracts semantic text chunks with structural context
2. **Render** -- PyMuPDF renders each page to PNG, uploaded to GCS for multimodal search
3. **Embed** -- Gemini Embedding 2 generates 768-dim vectors for both text chunks and page images
4. **Index** -- Vectors upserted to Vertex AI Vector Search (text index with TF-IDF hybrid, multimodal index for pages)

### Search (per query)

| Tool | Index | Method | Best for |
|------|-------|--------|----------|
| `search_text` | Text (hybrid) | Dense + TF-IDF sparse via RRF fusion | Facts, numbers, addresses |
| `search_pages` | Multimodal | Dense (image embeddings) | Tables, photos, maps, sketches |

The agent (Gemini 3.1 Pro) decides which tools to call, can search multiple rounds, and produces answers with inline citations linking to specific PDF pages.

### Frontend

- Server-Sent Events for live streaming of thinking, tool calls, and answer text
- Clickable inline citations that open a PDF viewer at the cited page
- Collapsible panel showing all retrieved documents grouped by source
- Google Material Design styling

## GCP Services Used

| Service | Purpose |
|---------|---------|
| **Cloud Storage** | PDF uploads, page images, embeddings, metadata |
| **Document AI** | Layout Parser for semantic PDF chunking |
| **Vertex AI Vector Search** | Two managed indexes (text hybrid + multimodal) |
| **Gemini 3.1 Pro** | Agentic reasoning with function calling |
| **Gemini Embedding 2** | Text and image embeddings (shared 768-dim space) |
| **Cloud Run** | Hosts the FastAPI app |
| **Eventarc** | Triggers pipeline on new GCS uploads |

## Project Structure

```
property_intelligence/
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
|   |-- create_indexes.py      # One-time: provision Vector Search indexes
|   `-- seed_data.py           # One-time: batch process initial PDFs
|-- .github/workflows/
|   `-- deploy.yml             # CI/CD: build, deploy to Cloud Run, create Eventarc trigger
|-- Dockerfile
|-- requirements.txt
`-- .env                       # Local secrets (gitignored)
```

## Setup

### Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI authenticated
- Python 3.11+
- APIs enabled:
  - Cloud Storage
  - Document AI
  - Vertex AI
  - Cloud Run
  - Eventarc

### 1. Clone and configure

```bash
git clone https://github.com/guanxi-shen/property-intelligence.git
cd property-intelligence
pip install -r requirements.txt
```

Create a `.env` file:

```env
GCP_PROJECT_ID=your-project-id
GCP_PROJECT_NUMBER=123456789
GOOGLE_CLOUD_REGION=us-central1

BUCKET_NAME=your-bucket-name

TEXT_SEARCH_API_ENDPOINT=
TEXT_SEARCH_INDEX_ENDPOINT=
TEXT_SEARCH_DEPLOYED_INDEX_ID=

MULTIMODAL_SEARCH_API_ENDPOINT=
MULTIMODAL_SEARCH_INDEX_ENDPOINT=
MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID=

DOCUMENT_AI_PROCESSOR_ID=your-processor-id
DOCUMENT_AI_LOCATION=us

credentials_dict={"type":"service_account", ...}
```

### 2. Create GCS bucket

```bash
gsutil mb -l us-central1 gs://your-bucket-name
```

### 3. Create Document AI processor

Go to **Cloud Console > Document AI > Create Processor** and select **Layout Parser**. Copy the processor ID into your `.env`.

### 4. Create Vector Search indexes

```bash
python scripts/create_indexes.py
```

This provisions two indexes (text + multimodal), creates endpoints, and deploys them. Copy the printed endpoint values into your `.env`. Index deployment takes ~30 minutes.

### 5. Upload PDFs and run pipeline

```bash
gsutil cp your_documents/*.pdf gs://your-bucket-name/uploads/
python scripts/seed_data.py
```

### 6. Run locally

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` in your browser.

## Deploy to Cloud Run

Deployment is automated via GitHub Actions on push to `main`.

### GitHub Secrets Required

Add these in **repo Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_PROJECT_NUMBER` | Google Cloud project number |
| `CREDENTIALS_DICT` | Service account JSON (used for both gcloud auth and runtime) |
| `BUCKET_NAME` | GCS bucket name |
| `TEXT_SEARCH_API_ENDPOINT` | Text index public endpoint domain |
| `TEXT_SEARCH_INDEX_ENDPOINT` | Text index full resource name |
| `TEXT_SEARCH_DEPLOYED_INDEX_ID` | Text deployed index ID |
| `MULTIMODAL_SEARCH_API_ENDPOINT` | Multimodal index public endpoint domain |
| `MULTIMODAL_SEARCH_INDEX_ENDPOINT` | Multimodal index full resource name |
| `MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID` | Multimodal deployed index ID |
| `DOCUMENT_AI_PROCESSOR_ID` | Document AI processor ID |
| `SERVICE_ACCOUNT_EMAIL` | Service account email for Eventarc |

Once secrets are configured, push to `main` to trigger deployment. The workflow builds the Docker image, deploys to Cloud Run, and creates an Eventarc trigger so new PDFs uploaded to GCS are processed automatically.

## License

[MIT](LICENSE)
