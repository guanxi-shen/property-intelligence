"""Property intelligence agent with dual-index search (text + page images)"""

import re
import logging
from typing import List, Dict, Any
from google.genai import types

from .config import AGENT_MODEL
from .llm import GeminiLLM
from .text_search import TextSearchClient
from .visual_search import PageSearchClient
from .prompts import SYSTEM_PROMPT, SEARCH_TEXT_DECLARATION, SEARCH_PAGE_IMG_DECLARATION

logger = logging.getLogger(__name__)

# Module-level search clients (initialized once)
_text_client = None
_page_client = None

# Module-level accumulators (reset per chat call)
_text_accumulator = {"results": [], "queries": [], "seen_ids": set(), "round": 0}
_page_accumulator = {"results": [], "queries": [], "seen_ids": set(), "round": 0}


def _get_text_client():
    global _text_client
    if _text_client is None:
        _text_client = TextSearchClient()
    return _text_client


def _get_page_client():
    global _page_client
    if _page_client is None:
        _page_client = PageSearchClient()
    return _page_client


def search_text(queries: list, neighbor_count: int = 15) -> dict:
    """Search text chunks from property documents with deduplication across rounds.

    Args:
        queries: List of search queries to execute.
        neighbor_count: Number of results per query.

    Returns:
        Dict with results, total_found, cumulative_total, round, message.
    """
    global _text_accumulator

    if not queries:
        return {"results": [], "total_found": 0, "message": "No queries provided"}

    _text_accumulator["round"] += 1
    current_round = _text_accumulator["round"]
    logger.info(f"[search_text] Round {current_round}, queries: {queries[:3]}")

    client = _get_text_client()
    batch_results = client.batch_search_and_retrieve(
        queries, neighbor_count=neighbor_count
    )

    # batch_search_and_retrieve returns a flat deduplicated list
    new_results = []
    for result in batch_results:
        doc_id = result.get("datapoint_id", "")
        if doc_id and doc_id not in _text_accumulator["seen_ids"]:
            _text_accumulator["seen_ids"].add(doc_id)
            entry = {
                "id": doc_id,
                "content": result.get("content", ""),
                "source_pdf": result.get("source_pdf", ""),
                "page_number": result.get("page_number", 0),
                "distance": result.get("distance", 0),
                "signed_url": result.get("signed_url", ""),
                "query_used": result.get("query_used", ""),
                "round": current_round,
            }
            _text_accumulator["results"].append(entry)
            new_results.append(entry)

    for query in queries:
        _text_accumulator["queries"].append({"query": query, "round": current_round})

    return {
        "results": new_results[:20],
        "total_found": len(new_results),
        "cumulative_total": len(_text_accumulator["results"]),
        "round": current_round,
        "message": f"Round {current_round}: Found {len(new_results)} new text chunks",
    }


def search_page_img(queries: list, neighbor_count: int = 8, diversity_balance: float = 0.5) -> dict:
    """Search PDF page images from property documents with deduplication across rounds.

    Args:
        queries: List of search queries for visual/page content.
        neighbor_count: Number of page results per query.
        diversity_balance: 0-1 controlling diversity across source PDFs.

    Returns:
        Dict with results (including gcs_uri for multimodal response),
        total_found, cumulative_total, round, message.
    """
    global _page_accumulator

    if not queries:
        return {"results": [], "total_found": 0, "message": "No queries provided"}

    _page_accumulator["round"] += 1
    current_round = _page_accumulator["round"]
    logger.info(f"[search_page_img] Round {current_round}, queries: {queries[:3]}")

    client = _get_page_client()
    batch_results = client.batch_search_page_img(
        queries, neighbor_count=neighbor_count, diversity_balance=diversity_balance
    )

    new_results = []
    for result in batch_results:
        doc_id = result.get("datapoint_id", "")
        if doc_id and doc_id not in _page_accumulator["seen_ids"]:
            _page_accumulator["seen_ids"].add(doc_id)
            entry = {
                "id": doc_id,
                "source_pdf": result.get("source_pdf", ""),
                "page_number": result.get("page_number", 0),
                "gcs_uri": result.get("gcs_uri", ""),
                "signed_url": result.get("signed_url", ""),
                "distance": result.get("distance", 0),
                "query_used": result.get("query_used", ""),
                "round": current_round,
            }
            _page_accumulator["results"].append(entry)
            new_results.append(entry)

    for query in queries:
        _page_accumulator["queries"].append({"query": query, "round": current_round})

    return {
        "results": new_results,
        "total_found": len(new_results),
        "cumulative_total": len(_page_accumulator["results"]),
        "round": current_round,
        "message": f"Round {current_round}: Found {len(new_results)} new page images",
    }


class PropertyAgent:
    """Single-agent property valuation chatbot with dual-index search tools."""

    def __init__(self):
        self.llm = GeminiLLM(
            model_name=AGENT_MODEL,
            system_instruction=SYSTEM_PROMPT,
            function_declarations=[SEARCH_TEXT_DECLARATION, SEARCH_PAGE_IMG_DECLARATION],
            function_handlers={
                "search_text": search_text,
                "search_page_img": search_page_img,
            },
        )
        self.conversation_history: List[tuple] = []

    def _reset_accumulators(self):
        global _text_accumulator, _page_accumulator
        _text_accumulator = {"results": [], "queries": [], "seen_ids": set(), "round": 0}
        _page_accumulator = {"results": [], "queries": [], "seen_ids": set(), "round": 0}

    def _parse_cited_sources(self, answer: str) -> List[Dict[str, Any]]:
        """Parse only the sources the agent explicitly cited in its answer.

        Matches markdown citation links like [source_name, p.N](url).
        Uses fuzzy name matching against accumulator for correct filenames.
        """
        citations = []
        seen = set()

        # Build lookups from accumulators
        # Exact: (source_pdf, page_number) -> metadata
        # Fuzzy: normalized_name -> actual source_pdf
        exact_lookup = {}
        name_map = {}
        for r in _text_accumulator["results"]:
            exact_lookup[(r["source_pdf"], r["page_number"])] = {**r, "type": "text"}
            norm = r["source_pdf"].lower().replace("_", " ").replace(".pdf", "")
            name_map[norm] = r["source_pdf"]
        for r in _page_accumulator["results"]:
            exact_lookup[(r["source_pdf"], r["page_number"])] = {**r, "type": "page_image"}
            norm = r["source_pdf"].lower().replace("_", " ").replace(".pdf", "")
            name_map[norm] = r["source_pdf"]

        # Parse [label, p.N](url) and [label, p.N] citations from answer
        for match in re.finditer(r'\[([^]]*?),\s*p\.?\s*(\d+)\](?:\(([^)]+)\))?', answer):
            label, page_str, url = match.group(1), match.group(2), match.group(3) or ""
            page = int(page_str)

            # Resolve actual filename via fuzzy match
            raw = label.strip()
            norm = raw.lower().replace("_", " ").replace(".pdf", "")
            source = name_map.get(norm, raw if raw.endswith('.pdf') else raw + '.pdf')

            key = (source, page)
            if key in seen:
                continue
            seen.add(key)

            meta = exact_lookup.get(key, {})
            citations.append({
                "source_pdf": source,
                "page_number": page,
                "signed_url": meta.get("signed_url", url),
                "type": meta.get("type", "text"),
            })

        return citations

    def _collect_retrieved_docs(self) -> List[Dict[str, Any]]:
        """Collect all unique retrieved documents from both accumulators."""
        docs = []
        seen = set()

        for r in _text_accumulator["results"]:
            key = (r["source_pdf"], r["page_number"])
            if key not in seen:
                seen.add(key)
                docs.append({
                    "source_pdf": r["source_pdf"],
                    "page_number": r["page_number"],
                    "content_preview": r.get("content", "")[:150],
                    "type": "text",
                })

        for r in _page_accumulator["results"]:
            key = (r["source_pdf"], r["page_number"])
            if key not in seen:
                seen.add(key)
                docs.append({
                    "source_pdf": r["source_pdf"],
                    "page_number": r["page_number"],
                    "content_preview": "",
                    "type": "page_image",
                })

        return docs

    def chat(self, message: str, stream_cb=None) -> Dict[str, Any]:
        """Process a user message, search as needed, and return a cited answer.

        Args:
            stream_cb: Optional callback(event_type, data) for live streaming to UI.

        Returns:
            Dict with 'answer', 'citations' (agent-cited), 'retrieved_docs' (all), 'thinking'.
        """
        self._reset_accumulators()

        contents = []
        for role, text in self.conversation_history:
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        ))

        result = self.llm.generate(prompt=message, contents=contents, stream_cb=stream_cb)

        answer = result.get("text", "")
        thinking = result.get("thinking", "")
        citations = self._parse_cited_sources(answer)
        retrieved_docs = self._collect_retrieved_docs()

        self.conversation_history.append(("user", message))
        self.conversation_history.append(("model", answer))

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_docs": retrieved_docs,
            "thinking": thinking,
        }
