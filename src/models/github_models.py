"""
GitHub Models adapter.

Uses the OpenAI-compatible inference endpoint provided by GitHub Models
(https://models.inference.ai.azure.com).  Requires a GitHub personal access
token (classic or fine-grained) stored as GITHUB_TOKEN in .env.

Free-tier limits apply (rate limits per minute/day vary by model).
"""

import asyncio
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from src.config import get_api_key, CONFIG
from src.models.base import LLMModel, CausalTriple, JudgmentResult, ExtractionError, TokenUsage
from src.prompts import format_extraction, format_complexity_judge, format_faithfulness_judge

logger = logging.getLogger(__name__)

_GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class GitHubModelsModel(LLMModel):
    """LLM adapter that calls GitHub Models via the OpenAI-compatible API."""

    def __init__(self, model_name: str | None = None):
        token = get_api_key("github")  # reads GITHUB_TOKEN from .env
        self._client = OpenAI(
            base_url=_GITHUB_MODELS_ENDPOINT,
            api_key=token,
        )
        self.model_name = model_name or CONFIG["models"].get("github", "gpt-4o-mini")
        self._executor = ThreadPoolExecutor(max_workers=20)  # Pool for async operations

    # ── internal helpers ─────────────────────────────────────────────────────

    def _call(self, system_prompt: str, user_prompt: str) -> tuple[str, TokenUsage]:
        """Call GitHub Models with retry / exponential backoff. Returns raw text and token usage."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Build API parameters
                api_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature": 0.0,
                }
                
                # Add max_completion_tokens: thinking models use max_thinking_tokens,
                # regular models use max_output_tokens (higher to avoid truncation)
                if any(x in self.model_name.lower() for x in ["o1", "o3", "thinking"]) and CONFIG.get("max_thinking_tokens"):
                    api_params["max_completion_tokens"] = CONFIG["max_thinking_tokens"]
                elif CONFIG.get("max_output_tokens"):
                    api_params["max_completion_tokens"] = CONFIG["max_output_tokens"]
                
                response = self._client.chat.completions.create(**api_params)
                raw = response.choices[0].message.content or ""
                
                # Extract token usage
                usage = TokenUsage()
                if hasattr(response, 'usage') and response.usage:
                    usage.prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    usage.completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                
                return raw, usage
            except Exception as exc:
                # Check if this is a rate limit error
                is_rate_limit = any(phrase in str(exc).lower() for phrase in 
                                   ["rate limit", "429", "too many requests", "quota"])
                
                if is_rate_limit:
                    # Longer backoff for rate limits
                    delay = (_BACKOFF_BASE ** attempt) * 2
                    logger.warning(
                        "GitHub Models RATE LIMIT hit (attempt %d/%d). Waiting %.1fs before retry...",
                        attempt, _MAX_RETRIES, delay
                    )
                else:
                    delay = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "GitHub Models attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
                    )
                
                if attempt < _MAX_RETRIES:
                    time.sleep(delay)
                else:
                    raise ExtractionError(f"GitHub Models failed after {_MAX_RETRIES} attempts: {exc}")
        raise ExtractionError(f"GitHub Models failed after {_MAX_RETRIES} attempts.")

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            cleaned = raw.strip()
            
            # Remove markdown code blocks
            if "```json" in cleaned:
                # Extract content between ```json and ```
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                if end > start:
                    cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                # Generic code block
                cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
            
            # If there's text before the JSON, try to find where JSON starts
            if cleaned and cleaned[0] not in '[{':
                # Look for first [ or {
                for i, char in enumerate(cleaned):
                    if char in '[{':
                        cleaned = cleaned[i:]
                        break
            
            return json.loads(cleaned)

    # ── public interface ─────────────────────────────────────────────────────

    def extract(self, passage: str) -> list[CausalTriple]:
        system, user = format_extraction(passage)
        raw, usage = self._call(system, user)
        try:
            data = self._parse_json(raw)
            # Unwrap the triples array if wrapped in an object
            items = data.get("triples", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = [items]
            
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
