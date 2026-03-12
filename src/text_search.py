"""Text search client using hybrid dense+sparse retrieval"""

import json
import numpy as np
import scipy.sparse as sp
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from google.cloud import storage
from google.cloud import aiplatform_v1
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchingEngineIndexEndpoint, HybridQuery
)
import vertexai

from .config import (
    PROJECT_ID, CREDENTIALS, BUCKET_NAME,
    TEXT_SEARCH_API_ENDPOINT, TEXT_SEARCH_INDEX_ENDPOINT,
    TEXT_SEARCH_DEPLOYED_INDEX_ID, TEXT_METADATA_PATH,
    SPARSE_VECTORIZER_PATH, EMBEDDING_MODEL, EMBEDDING_DIM,
    USE_HYBRID_SEARCH, RRF_ALPHA,
)


class TextSearchClient:
    """Hybrid dense+sparse search over property document text chunks"""

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

        # Load metadata and sparse vectorizer
        print(f"[TextSearch] Loading metadata from {TEXT_METADATA_PATH}")
        self.metadata_lookup = self._load_metadata()
        print(f"[TextSearch] Loaded {len(self.metadata_lookup)} datapoints")

        if USE_HYBRID_SEARCH:
            print(f"[TextSearch] Loading sparse vectorizer from {SPARSE_VECTORIZER_PATH}")
            self.sparse_vectorizer = self._load_sparse_vectorizer()
            print(f"[TextSearch] Vectorizer loaded: {len(self.sparse_vectorizer.vocabulary_)} terms")
        else:
            self.sparse_vectorizer = None

    def _load_metadata(self):
        """Load chunk metadata lookup from GCS"""
        try:
            blob = self.bucket.blob(TEXT_METADATA_PATH)
            return json.loads(blob.download_as_text())
        except Exception as e:
            print(f"[TextSearch] Error loading metadata: {e}")
            return {}

    def _load_sparse_vectorizer(self):
        """Reconstruct TF-IDF vectorizer from saved state in GCS"""
        blob = self.bucket.blob(SPARSE_VECTORIZER_PATH)
        if not blob.exists():
            raise FileNotFoundError(
                f"Sparse vectorizer not found at gs://{BUCKET_NAME}/{SPARSE_VECTORIZER_PATH}. "
                f"Run embedding pipeline first or set USE_HYBRID_SEARCH=False."
            )

        state = json.loads(blob.download_as_text())
        vec = TfidfVectorizer(ngram_range=tuple(state['ngram_range']))
        vec.vocabulary_ = state['vocabulary']
        vec.idf_ = np.array(state['idf'])
        vec._tfidf._idf_diag = sp.spdiags(vec.idf_, 0, len(vec.idf_), len(vec.idf_))
        return vec

    def _get_query_embedding(self, query_text):
        """Generate dense embedding for a search query via Gemini Embedding 2"""
        response = self.embed_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query_text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return response.embeddings[0].values

    def _get_sparse_embedding(self, text):
        """Generate sparse TF-IDF embedding for a query"""
        tfidf_vector = self.sparse_vectorizer.transform([text])
        values = [float(v) for v in tfidf_vector.data]
        dims = [int(d) for d in tfidf_vector.indices]
        return {"values": values, "dimensions": dims}

    def generate_signed_url(self, gcs_uri, expiration_days=7):
        """Generate a signed URL for secure document access"""
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
            print(f"[TextSearch] Signed URL error for {gcs_uri}: {e}")
            return ""

    def batch_search_and_retrieve(self, queries, neighbor_count=15):
        """Search text index for multiple queries, return flat list of results.

        Uses hybrid search (dense + sparse with RRF) when enabled, otherwise dense-only.
        Queries run in parallel via ThreadPoolExecutor.

        Returns:
            List of result dicts, each with:
            {datapoint_id, source_pdf, page_number, chunk_type, content,
             distance, signed_url, query_used}
        """
        if not queries:
            return []

        if USE_HYBRID_SEARCH and self.sparse_vectorizer:
            return self._batch_search_hybrid(queries, neighbor_count)
        return self._batch_search_dense(queries, neighbor_count)

    # -- Hybrid search (dense + sparse, RRF fusion) --

    def _batch_search_hybrid(self, queries, neighbor_count):
        """Parallel hybrid search over all queries"""
        # Generate dense embeddings for all queries
        dense_embeddings = [self._get_query_embedding(q) for q in queries]
        sparse_embeddings = [self._get_sparse_embedding(q) for q in queries]

        # Build HybridQuery objects
        hybrid_queries = [
            HybridQuery(
                dense_embedding=dense,
                sparse_embedding_dimensions=sparse["dimensions"],
                sparse_embedding_values=sparse["values"],
                rrf_ranking_alpha=RRF_ALPHA,
            )
            for dense, sparse in zip(dense_embeddings, sparse_embeddings)
        ]

        # MatchingEngineIndexEndpoint requires vertexai init
        vertexai.init(
            project=PROJECT_ID, location='us-central1', credentials=CREDENTIALS
        )
        endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=TEXT_SEARCH_INDEX_ENDPOINT
        )

        def _search_one(args):
            idx, hq, query_text = args
            try:
                response = endpoint.find_neighbors(
                    deployed_index_id=TEXT_SEARCH_DEPLOYED_INDEX_ID,
                    queries=[hq],
                    num_neighbors=neighbor_count,
                )
                neighbors = response[0] if response else []
                return idx, neighbors, query_text, None
            except Exception as e:
                return idx, [], query_text, str(e)

        # Parallel execution
        results_map = {}
        work = [
            (i, hq, qt)
            for i, (hq, qt) in enumerate(zip(hybrid_queries, queries))
        ]
        with ThreadPoolExecutor(max_workers=min(len(work), 10)) as pool:
            for future in as_completed(pool.submit(_search_one, w) for w in work):
                idx, neighbors, query_text, error = future.result()
                if error:
                    print(f"[TextSearch] Query {idx} failed: {error[:120]}")
                results_map[idx] = (neighbors, query_text)

        # Parse results in original query order
        return self._parse_hybrid_results(results_map, len(queries))

    def _parse_hybrid_results(self, results_map, num_queries):
        """Convert hybrid search neighbors into result dicts"""
        all_results = []
        seen_ids = set()

        for i in range(num_queries):
            neighbors, query_text = results_map.get(i, ([], ""))
            for neighbor in neighbors:
                dp_id = neighbor.id
                if dp_id in seen_ids:
                    continue
                seen_ids.add(dp_id)

                meta = self.metadata_lookup.get(dp_id, {})
                all_results.append({
                    'datapoint_id': dp_id,
                    'source_pdf': meta.get('source_pdf', 'Unknown'),
                    'page_number': meta.get('page_number', 'Unknown'),
                    'chunk_type': meta.get('chunk_type', 'Unknown'),
                    'content': meta.get('content', ''),
                    'distance': neighbor.distance,
                    'signed_url': '',
                    'gcs_uri': meta.get('gcs_uri', ''),
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

    # -- Dense-only fallback --

    def _batch_search_dense(self, queries, neighbor_count):
        """Dense-only batch search using low-level gRPC client"""
        embeddings = [self._get_query_embedding(q) for q in queries]

        client = aiplatform_v1.MatchServiceClient(
            client_options={"api_endpoint": TEXT_SEARCH_API_ENDPOINT},
            credentials=CREDENTIALS,
        )

        vs_queries = []
        for emb in embeddings:
            dp = aiplatform_v1.IndexDatapoint(feature_vector=emb)
            vs_queries.append(
                aiplatform_v1.FindNeighborsRequest.Query(
                    datapoint=dp, neighbor_count=neighbor_count
                )
            )

        response = client.find_neighbors(
            aiplatform_v1.FindNeighborsRequest(
                index_endpoint=TEXT_SEARCH_INDEX_ENDPOINT,
                deployed_index_id=TEXT_SEARCH_DEPLOYED_INDEX_ID,
                queries=vs_queries,
                return_full_datapoint=False,
            )
        )

        return self._parse_dense_results(response, queries)

    def _parse_dense_results(self, response, queries):
        """Convert dense search response into result dicts"""
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
                    'chunk_type': meta.get('chunk_type', 'Unknown'),
                    'content': meta.get('content', ''),
                    'distance': neighbor.distance,
                    'signed_url': '',
                    'gcs_uri': meta.get('gcs_uri', ''),
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


