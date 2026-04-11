"""OpenAI adapter (fallback provider)."""

import asyncio
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, APIError, RateLimitError

from src.config import get_api_key, CONFIG
from src.models.base import LLMModel, CausalTriple, JudgmentResult, ExtractionError, TokenUsage
from src.prompts import format_extraction, format_complexity_judge, format_faithfulness_judge

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class OpenAIModel(LLMModel):
    def __init__(self, model_name: str | None = None):
        api_key = get_api_key("openai")
        self._client = OpenAI(api_key=api_key)
        self.model_name = model_name or CONFIG["models"]["openai"]
        self._executor = ThreadPoolExecutor(max_workers=20)  # Pool for async operations

    # ── internal helpers ─────────────────────────────────────────────────────

    def _call(self, system_prompt: str, user_prompt: str) -> tuple[str, TokenUsage]:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Build API parameters
                api_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                
                # Add max_completion_tokens: thinking models (o1/o3) use max_thinking_tokens,
                # regular models use max_output_tokens (higher to avoid truncation)
                if any(x in self.model_name.lower() for x in ["o1", "o3", "thinking"]) and CONFIG.get("max_thinking_tokens"):
                    api_params["max_completion_tokens"] = CONFIG["max_thinking_tokens"]
                elif CONFIG.get("max_output_tokens"):
                    api_params["max_completion_tokens"] = CONFIG["max_output_tokens"]
                
                response = self._client.chat.completions.create(**api_params)
                
                # Extract token usage
                usage = TokenUsage()
                if hasattr(response, 'usage') and response.usage:
                    usage.prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    usage.completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                
                return response.choices[0].message.content, usage
            except RateLimitError as exc:
                logger.warning("OpenAI rate limit (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_BACKOFF_BASE ** attempt)
            except APIError as exc:
                logger.warning("OpenAI API error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_BACKOFF_BASE ** attempt)
        raise ExtractionError(f"OpenAI failed after {_MAX_RETRIES} attempts.")

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)

    # ── public interface ─────────────────────────────────────────────────────

    def extract(self, passage: str) -> list[CausalTriple]:
        system, user = format_extraction(passage)
        # OpenAI's json_object mode requires the array to be wrapped; unwrap it here.
        wrapped_user = user + "\n\nWrap the JSON array in an object: {\"triples\": [...]}"
        raw, usage = self._call(system, wrapped_user)
        try:
            data = self._parse_json(raw)
            items = data.get("triples", data) if isinstance(data, dict) else data
            triples = []
            for item in items:
                cause = item.get("cause", "")
                effect = item.get("effect", "")
                
                # Filter out invalid triples where cause or effect is "none" or empty
                if not cause or cause.lower() == "none" or not effect or effect.lower() == "none":
                    logger.debug("Skipping invalid triple with cause='%s' or effect='%s'", cause, effect)
                    continue
                
                triples.append(CausalTriple(
                    cause=cause,
                    connector=item.get("connector", ""),
                    effect=effect,
                    hedge=item.get("hedge", ""),
                    direction=item.get("direction", "ambiguous"),
                    raw_response=raw,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                ))
            return triples
        except Exception as exc:
            logger.error("Failed to parse extraction response: %s\nRaw: %s", exc, raw)
            return []

    async def extract_async(self, passage: str) -> list[CausalTriple]:
        """
        Async version of extract() for concurrent batch processing.
        
        Runs the synchronous extract() method in a thread pool to avoid
        blocking the event loop while maintaining the same extraction logic.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.extract, passage)

    async def extract_async(self, passage: str) -> list[CausalTriple]:
        """Async version of extract() for concurrent batch processing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.extract, passage)

    def judge(self, passage: str, triple: CausalTriple) -> JudgmentResult:
        # Complexity
        sys_c, usr_c = format_complexity_judge(passage)
        raw_c, usage_c = self._call(sys_c, usr_c)
        try:
            comp = self._parse_json(raw_c)
        except Exception as exc:
            logger.error("Failed to parse complexity response: %s\nRaw: %s", exc, raw_c)
            comp = {"complexity_score": 0, "complexity_rationale": "parse error"}

        # Faithfulness
        sys_f, usr_f = format_faithfulness_judge(
            passage, triple.cause, triple.connector,
            triple.effect, triple.hedge, triple.direction,
        )
        raw_f, usage_f = self._call(sys_f, usr_f)
        try:
            faith = self._parse_json(raw_f)
        except Exception as exc:
            logger.error("Failed to parse faithfulness response: %s\nRaw: %s", exc, raw_f)
            faith = {"faithful": -1, "faithful_rationale": "parse error", "failure_mode": ""}

        return JudgmentResult(
            complexity_score=int(comp.get("complexity_score", 0)),
            faithful=int(faith.get("faithful", -1)),
            failure_mode=faith.get("failure_mode", ""),
        )
