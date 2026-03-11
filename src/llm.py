"""Gemini LLM with function calling and multimodal support via Vertex AI.

FC loop pattern adapted from freetitle_ai_studio: manual execution with
parallel tool dispatch and thought signature preservation."""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from .config import PROJECT_ID, CREDENTIALS, AGENT_MODEL, MAX_FUNCTION_CALLING_ROUNDS

STREAM_MAX_RETRIES = 4

logger = logging.getLogger(__name__)

SAFETY_SETTINGS_NONE = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
]


class GeminiLLM:
    """Gemini LLM with iterative function calling and multimodal response support."""

    def __init__(
        self,
        model_name: str = AGENT_MODEL,
        system_instruction: str = None,
        function_declarations: list = None,
        function_handlers: dict = None,
        thinking_level: str = "medium",
        max_function_calling_rounds: int = MAX_FUNCTION_CALLING_ROUNDS,
    ):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.function_declarations = function_declarations
        self.function_handlers = function_handlers or {}
        self.thinking_level = thinking_level
        self.max_function_calling_rounds = max_function_calling_rounds
        self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location="global",
                credentials=CREDENTIALS,
            )
        return self._client

    # -- Streaming response with part accumulation --

    def _stream_response(self, contents, config, stream_cb=None):
        """Stream a single LLM call with 429 retry.

        Args:
            stream_cb: Optional callback(event_type, data) for live streaming.

        Returns (text, thinking, function_calls, accumulated_parts).
        """
        client = self._get_client()

        for attempt in range(STREAM_MAX_RETRIES):
            try:
                text, thinking = "", ""
                function_calls = []
                accumulated_parts = []

                for chunk in client.models.generate_content_stream(
                    model=self.model_name, contents=contents, config=config
                ):
                    if not (chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts):
                        continue
                    for part in chunk.candidates[0].content.parts:
                        accumulated_parts.append(part)
                        is_thought = getattr(part, 'thought', False)
                        has_text = hasattr(part, 'text') and part.text
                        has_fc = hasattr(part, 'function_call') and part.function_call is not None

                        if is_thought and has_text:
                            thinking += part.text
                            if stream_cb:
                                stream_cb("thinking", part.text)
                        elif has_text and not has_fc:
                            text += part.text
                            if stream_cb:
                                stream_cb("text", part.text)

                        if has_fc:
                            function_calls.append(part.function_call)

                return text, thinking, function_calls, accumulated_parts

            except Exception as e:
                if attempt < STREAM_MAX_RETRIES - 1 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                    wait = 2 ** attempt * 5
                    logger.warning(f"429 rate limit, retry {attempt + 1}/{STREAM_MAX_RETRIES} in {wait}s")
                    time.sleep(wait)
                    continue
                raise

    # -- Content normalization --

    def _normalize_contents(self, contents):
        """Fix role alternation: insert model ack between consecutive user-role items."""
        if not contents:
            return contents

        result = []
        for item in contents:
            is_user = isinstance(item, types.Content) and item.role == 'user'
            if is_user and result:
                prev = result[-1]
                prev_is_user = isinstance(prev, types.Content) and prev.role == 'user'
                if prev_is_user:
                    result.append(types.Content(
                        role='model',
                        parts=[types.Part.from_text(text='Understood.')],
                    ))
            result.append(item)
        return result

    # -- Function calling --

    def _execute_tools(self, function_calls):
        """Execute function calls in parallel. Returns list of (fc, result) tuples."""
        results = [None] * len(function_calls)

        with ThreadPoolExecutor(max_workers=min(len(function_calls), 8)) as pool:
            future_to_idx = {}
            for i, fc in enumerate(function_calls):
                handler = self.function_handlers.get(fc.name)
                if handler:
                    args = dict(fc.args) if hasattr(fc, 'args') else {}
                    future = pool.submit(handler, **args)
                    future_to_idx[future] = i
                else:
                    results[i] = (fc, {"error": f"Unknown function: {fc.name}"})

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"error": str(e)}
                results[idx] = (function_calls[idx], result)

        return results

    def _build_function_response_parts(self, fc, result):
        """Build function response Part(s), attaching page images via FunctionResponseFileData.

        For search_page_img results containing gcs_uri, each page image is attached
        as a FunctionResponsePart with file_data, and referenced in the JSON via
        {"$ref": "display_name"} so the model can map metadata to visual content.

        Page images are stored at: gs://{bucket}/page_images/{pdf_stem}/page_{N}.png
        Display names use the datapoint_id pattern: {pdf_stem}_page_{N}.png
        """
        has_images = isinstance(result, dict) and any(
            r.get("gcs_uri") for r in result.get("results", [])
        )

        if not has_images:
            return [types.Part.from_function_response(name=fc.name, response=result)]

        # Build response data with $ref links to attached file parts
        items = []
        file_parts = []

        for item in result.get("results", []):
            gcs_uri = item.get("gcs_uri", "")
            # Display name ties the JSON item to its file_data part
            display_name = f"{item.get('id', '')}.png" if item.get("id") else gcs_uri.split("/")[-1]

            item_data = {
                "id": item.get("id", ""),
                "source_pdf": item.get("source_pdf", ""),
                "page_number": item.get("page_number", 0),
                "signed_url": item.get("signed_url", ""),
            }

            if gcs_uri:
                # JSON reference to the attached file_data part
                item_data["page_image"] = {"$ref": display_name}
                try:
                    file_parts.append(types.FunctionResponsePart(
                        file_data=types.FunctionResponseFileData(
                            mime_type="image/png",
                            display_name=display_name,
                            file_uri=gcs_uri,
                        )
                    ))
                except Exception as e:
                    logger.warning(f"Could not attach page image {gcs_uri}: {e}")

            items.append(item_data)

        response_data = {
            "total_found": result.get("total_found", 0),
            "cumulative_total": result.get("cumulative_total", 0),
            "round": result.get("round", 1),
            "message": result.get("message", ""),
            "items": items,
        }

        if file_parts:
            return [types.Part.from_function_response(
                name=fc.name, response=response_data, parts=file_parts
            )]
        return [types.Part.from_function_response(name=fc.name, response=response_data)]

    # -- FC loop --

    def _fc_loop(self, contents, config, stream_cb=None):
        """Function calling loop with parallel tool execution and thought signature preservation."""
        contents = list(contents)
        all_thinking = ""
        text = ""

        for round_num in range(self.max_function_calling_rounds):
            logger.info(f"FC round {round_num + 1}")

            text, thinking, function_calls, accumulated_parts = self._stream_response(
                contents, config, stream_cb
            )

            if thinking:
                first_line = thinking.strip().split('\n')[0][:200]
                all_thinking += f"\n[Round {round_num + 1}] {first_line}"

            if not function_calls:
                break

            # Emit tool call events
            if stream_cb:
                for fc in function_calls:
                    stream_cb("tool_call", fc.name)

            contents.append(types.Content(role="model", parts=accumulated_parts))
            results = self._execute_tools(function_calls)

            # Emit tool result events
            if stream_cb:
                for fc, result in results:
                    msg = result.get("message", "") if isinstance(result, dict) else ""
                    stream_cb("tool_result", msg)

            response_parts = []
            for fc, result in results:
                parts = self._build_function_response_parts(fc, result)
                response_parts.extend(parts)

            contents.append(types.Content(role="tool", parts=response_parts))

        return {
            "thinking": all_thinking.strip(),
            "text": text or "No response generated",
        }

    # -- Main entry point --

    def generate(self, prompt: str, contents: list = None, stream_cb=None) -> dict:
        """Generate a response, handling function calling and multimodal input.

        Args:
            prompt: Text prompt (used when contents is None).
            contents: Pre-built contents list for multi-turn conversation.

        Returns:
            dict with 'thinking' and 'text'.
        """
        try:
            if contents is None:
                contents = [types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )]

            contents = self._normalize_contents(contents)

            config_kwargs = {
                "system_instruction": self.system_instruction,
                "safety_settings": SAFETY_SETTINGS_NONE,
                "thinking_config": types.ThinkingConfig(
                    thinking_level=self.thinking_level,
                    include_thoughts=True,
                ),
            }

            if self.function_declarations:
                config_kwargs["tools"] = [types.Tool(
                    function_declarations=self.function_declarations
                )]
                config_kwargs["automatic_function_calling"] = (
                    types.AutomaticFunctionCallingConfig(disable=True)
                )

            config = types.GenerateContentConfig(**config_kwargs)

            if self.function_declarations:
                return self._fc_loop(contents, config, stream_cb)

            # No tools -- single call
            text, thinking, _, _ = self._stream_response(contents, config, stream_cb)
            return {
                "thinking": thinking,
                "text": text,
            }

        except Exception as e:
            logger.error(f"Gemini error: {e}", exc_info=True)
            return {
                "thinking": "",
                "text": f"LLM error: {e}",
            }
