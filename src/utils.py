"""Utility functions for the RAG system"""

import json
import re
from typing import Any, Optional, Union


def parse_llm_json(response_text: str, fallback: Any = None) -> Optional[Union[dict, list, Any]]:
    """Parse JSON from LLM responses that may be wrapped in code blocks or mixed with text."""
    if not response_text or not isinstance(response_text, str):
        return fallback

    cleaned = response_text.strip()

    # Try ```json code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try JSON array or object pattern
    match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try raw parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return fallback


def clean_json_response(response_text: str) -> str:
    """Strip markdown code block wrappers from JSON strings."""
    if response_text.startswith("```json"):
        response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
    return response_text.strip()
