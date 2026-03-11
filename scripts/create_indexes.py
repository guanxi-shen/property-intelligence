"""One-time script to create Vertex AI Vector Search indexes and endpoints.

Run once to provision the text and multimodal indexes, then copy the
printed endpoint IDs and API endpoints into your .env file.

Before running: upload at least one dummy JSONL embedding file to the
embeddings paths so the index has initial data to build from.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.cloud import aiplatform, storage
from src.config import (
    PROJECT_ID, LOCATION, CREDENTIALS, EMBEDDING_DIM, BUCKET_NAME,
    TEXT_EMBEDDINGS_PATH, MULTIMODAL_EMBEDDINGS_PATH,
)

aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=CREDENTIALS)


def _ensure_seed_embedding(gcs_path: str):
    """Upload a single dummy embedding so the index has valid initial data."""
    client = storage.Client(credentials=CREDENTIALS, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    if blob.exists():
        print(f"  Seed data already exists at gs://{BUCKET_NAME}/{gcs_path}")
        return

    dummy = {
        "id": "seed_0",
        "embedding": [0.0] * EMBEDDING_DIM,
    }
    blob.upload_from_string(json.dumps(dummy) + "\n", content_type="application/json")
    print(f"  Uploaded seed embedding to gs://{BUCKET_NAME}/{gcs_path}")


def create_text_index():
    """Create text index with hybrid search (dense + sparse) support."""
    print("\n--- Text Index ---")
    _ensure_seed_embedding(TEXT_EMBEDDINGS_PATH)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="property-text-index",
        description="Hybrid dense+sparse text chunk embeddings from property documents",
        contents_delta_uri=f"gs://{BUCKET_NAME}/embeddings/text/",
        dimensions=EMBEDDING_DIM,
        approximate_neighbors_count=100,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        feature_norm_type="UNIT_L2_NORM",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=50,
        index_update_method="BATCH_UPDATE",
        shard_size="SHARD_SIZE_SMALL",
    )
    print(f"  Index created: {index.resource_name}")
    return index


def create_multimodal_index():
    """Create multimodal index for page image embeddings (dense only)."""
    print("\n--- Multimodal Index ---")
    _ensure_seed_embedding(MULTIMODAL_EMBEDDINGS_PATH)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="property-multimodal-index",
        description="Page image embeddings from property documents",
        contents_delta_uri=f"gs://{BUCKET_NAME}/embeddings/multimodal/",
        dimensions=EMBEDDING_DIM,
        approximate_neighbors_count=50,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        feature_norm_type="UNIT_L2_NORM",
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=20,
        index_update_method="BATCH_UPDATE",
        shard_size="SHARD_SIZE_SMALL",
    )
    print(f"  Index created: {index.resource_name}")
    return index


def create_endpoint(display_name: str):
    """Create a public Vector Search index endpoint."""
    print(f"\nCreating endpoint: {display_name}")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=display_name,
        public_endpoint_enabled=True,
    )
    print(f"  Endpoint created: {endpoint.resource_name}")
    print(f"  Public domain: {endpoint.public_endpoint_domain_name}")
    return endpoint


def deploy_index(endpoint, index, deployed_index_id: str):
    """Deploy an index to an endpoint."""
    print(f"\nDeploying {deployed_index_id}...")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        display_name=deployed_index_id,
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1,
    )
    print(f"  Deployed: {deployed_index_id}")


def main():
    # -- Text index (hybrid dense+sparse) --
    text_index = create_text_index()
    text_endpoint = create_endpoint("property-text-endpoint")
    deploy_index(text_endpoint, text_index, "property_text_deployed")

    # -- Multimodal index (dense only, page images) --
    mm_index = create_multimodal_index()
    mm_endpoint = create_endpoint("property-multimodal-endpoint")
    deploy_index(mm_endpoint, mm_index, "property_multimodal_deployed")

    # -- Print values for .env --
    print("\n" + "=" * 60)
    print("Add these to your .env file:")
    print("=" * 60)
    print(f"TEXT_SEARCH_INDEX_ENDPOINT={text_endpoint.resource_name}")
    print(f"TEXT_SEARCH_API_ENDPOINT={text_endpoint.public_endpoint_domain_name}")
    print(f"TEXT_SEARCH_DEPLOYED_INDEX_ID=property_text_deployed")
    print()
    print(f"MULTIMODAL_SEARCH_INDEX_ENDPOINT={mm_endpoint.resource_name}")
    print(f"MULTIMODAL_SEARCH_API_ENDPOINT={mm_endpoint.public_endpoint_domain_name}")
    print(f"MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID=property_multimodal_deployed")
    print("=" * 60)


if __name__ == "__main__":
    main()
