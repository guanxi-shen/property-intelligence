"""Configuration for Property Intelligence RAG system"""

import os
import json
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

# GCP
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
LOCATION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')

# Storage
BUCKET_NAME = os.getenv('BUCKET_NAME', 'property-intelligence')
UPLOADS_PREFIX = 'uploads/'
PROCESSED_PREFIX = 'processed/'
PAGE_IMAGES_PREFIX = 'page_images/'

# Text Vector Search
TEXT_SEARCH_API_ENDPOINT = os.getenv('TEXT_SEARCH_API_ENDPOINT')
TEXT_SEARCH_INDEX_ENDPOINT = os.getenv('TEXT_SEARCH_INDEX_ENDPOINT')
TEXT_SEARCH_DEPLOYED_INDEX_ID = os.getenv('TEXT_SEARCH_DEPLOYED_INDEX_ID')

# Multimodal Vector Search (page images)
MULTIMODAL_SEARCH_API_ENDPOINT = os.getenv('MULTIMODAL_SEARCH_API_ENDPOINT')
MULTIMODAL_SEARCH_INDEX_ENDPOINT = os.getenv('MULTIMODAL_SEARCH_INDEX_ENDPOINT')
MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID = os.getenv('MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID')

# Document AI
DOCUMENT_AI_PROCESSOR_ID = os.getenv('DOCUMENT_AI_PROCESSOR_ID')
DOCUMENT_AI_LOCATION = os.getenv('DOCUMENT_AI_LOCATION', 'us')

# Project number (for Document AI processor paths)
PROJECT_NUMBER = os.getenv('GCP_PROJECT_NUMBER', '')

# Embedding
EMBEDDING_MODEL = 'gemini-embedding-2-preview'
EMBEDDING_DIM = 768  # Matryoshka: 768, 1536, or 3072

# Hybrid Search
USE_HYBRID_SEARCH = True
RRF_ALPHA = 0.85  # 85% dense, 15% sparse

# Metadata paths in GCS
TEXT_METADATA_PATH = 'metadata/text/datapoint_lookup.json'
MULTIMODAL_METADATA_PATH = 'metadata/multimodal/datapoint_lookup.json'
SPARSE_VECTORIZER_PATH = 'metadata/text/sparse_vectorizer_state.json'
TEXT_EMBEDDINGS_PATH = 'embeddings/text/property_embeddings.json'
MULTIMODAL_EMBEDDINGS_PATH = 'embeddings/multimodal/property_page_embeddings.json'

# Agent
AGENT_MODEL = 'gemini-3.1-pro-preview'
MAX_FUNCTION_CALLING_ROUNDS = 5

# Batch processing
TEXT_EMBED_BATCH_SIZE = 250  # max per embed_content() call
IMAGE_EMBED_WORKERS = 10    # parallel threads for image embedding


def initialize_credentials():
    """Initialize GCP credentials from service account JSON in env"""
    credentials_json = os.getenv('credentials_dict')
    if credentials_json:
        credentials_info = json.loads(credentials_json)
        return service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
    # Fall back to application default credentials
    import google.auth
    credentials, _ = google.auth.default()
    return credentials


CREDENTIALS = initialize_credentials()
