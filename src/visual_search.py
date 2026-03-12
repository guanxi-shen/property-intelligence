"""Page image search client using Gemini Embedding 2 multimodal index"""

import json
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from google.cloud import storage
from google.cloud import aiplatform_v1

from .config import (
    PROJECT_ID, CREDENTIALS, BUCKET_NAME,
    MULTIMODAL_SEARCH_API_ENDPOINT, MULTIMODAL_SEARCH_INDEX_ENDPOINT,
    MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID, MULTIMODAL_METADATA_PATH,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)


class PageSearchClient:
    """Dense search over rendered PDF page images.

    Uses the same Gemini Embedding 2 model as text search, so text queries
    naturally match page images in the shared semantic space.
    Crowding tag = source PDF filename for cross-document diversity.
    """

    def __init__(self):
        self.storage_client = storage.Client(
            credentials=CREDENTIALS, project=PROJECT_ID
        )
        self.bucket = self.storage_client.bucket(BUCKET_NAME)

        # Gemini Embedding 2 client
        self.embed_client = genai.Client(
            vertexai=True, project=PROJECT_ID,
            location="us-central1", credentials=CREDENTIALS,
        )

        # Load page metadata
        print(f"[PageSearch] Loading metadata from {MULTIMODAL_METADATA_PATH}")
        self.metadata_lookup = self._load_metadata()
        print(f"[PageSearch] Loaded {len(self.metadata_lookup)} page entries")

    def _load_metadata(self):
        """Load page image metadata lookup from GCS"""
        try:
            blob = self.bucket.blob(MULTIMODAL_METADATA_PATH)
            return json.loads(blob.download_as_text())
        except Exception as e:
            print(f"[PageSearch] Error loading metadata: {e}")
            return {}

    def _get_query_embedding(self, query_text):
        """Generate dense embedding for a text query via Gemini Embedding 2.

        Text queries search against page image embeddings -- both live in
        the same semantic space because they use the same model.
        """
        response = self.embed_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query_text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return response.embeddings[0].values

    def generate_signed_url(self, gcs_uri, expiration_days=7):
        """Generate a signed URL for secure page image access"""
        try:
            if not gcs_uri or not gcs_uri.startswith('gs://'):
                return ""
            path_parts = gcs_uri.replace('gs://', '').split('/', 1)
            if len(path_parts) != 2:
                return ""

            bucket = self.storage_client.bucket(path_parts[0])
            blob = bucket.blob(path_parts[1])
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(days=expiration_days),
                method="GET",
            )
        except Exception as e:
            print(f"[PageSearch] Signed URL error for {gcs_uri}: {e}")
            return ""

    def batch_search_page_img(self, queries, neighbor_count=8, diversity_balance=0.5):
        """Search page image index for multiple queries.

        Crowding tag limits results per source PDF so diverse documents
        are represented in results.

        Args:
            queries: Text queries to search for visually relevant pages.
            neighbor_count: Pages to return per query.
            diversity_balance: Max fraction of results from one PDF (0-1).

        Returns:
            Flat deduplicated list of result dicts, each with:
            {datapoint_id, source_pdf, page_number, gcs_uri,
             distance, signed_url, query_used}
        """
        if not queries:
            return []

        # Generate embeddings for all queries
        query_embeddings = [self._get_query_embedding(q) for q in queries]

        # Crowding limits pages per source document
        max_per_doc = max(1, int(neighbor_count * diversity_balance)) if diversity_balance else None

        # Build Vector Search request
        client = aiplatform_v1.MatchServiceClient(
            client_options={"api_endpoint": MULTIMODAL_SEARCH_API_ENDPOINT},
            credentials=CREDENTIALS,
        )

        vs_queries = []
        for emb in query_embeddings:
            dp = aiplatform_v1.IndexDatapoint(feature_vector=emb)
            vs_queries.append(
                aiplatform_v1.FindNeighborsRequest.Query(
                    datapoint=dp,
                    neighbor_count=neighbor_count,
                    per_crowding_attribute_neighbor_count=max_per_doc,
                )
            )

        response = client.find_neighbors(
            aiplatform_v1.FindNeighborsRequest(
                index_endpoint=MULTIMODAL_SEARCH_INDEX_ENDPOINT,
                deployed_index_id=MULTIMODAL_SEARCH_DEPLOYED_INDEX_ID,
                queries=vs_queries,
                return_full_datapoint=False,
            )
        )

        return self._parse_results(response, queries)

    def _parse_results(self, response, queries):
        """Convert search response into deduplicated result dicts"""
        all_results = []
        seen_ids = set()

        for i, query_text in enumerate(queries):
            if not response.nearest_neighbors or i >= len(response.nearest_neighbors):
                continue
            for neighbor in response.nearest_neighbors[i].neighbors:
                dp_id = neighbor.datapoint.datapoint_id
                if dp_id in seen_ids:
                    continue
                seen_ids.add(dp_id)

                meta = self.metadata_lookup.get(dp_id, {})
                all_results.append({
                    'datapoint_id': dp_id,
                    'source_pdf': meta.get('source_pdf', 'Unknown'),
                    'page_number': meta.get('page_number', 'Unknown'),
                    'gcs_uri': meta.get('gcs_uri', ''),
                    'distance': neighbor.distance,
                    'signed_url': '',
                    'query_used': query_text,
                })

        all_results.sort(key=lambda x: x['distance'])

        # Generate signed URLs in parallel
        gcs_uris = [r['gcs_uri'] for r in all_results]
        with ThreadPoolExecutor(max_workers=8) as pool:
            urls = list(pool.map(self.generate_signed_url, gcs_uris))
        for r, url in zip(all_results, urls):
            r['signed_url'] = url

        return all_results


