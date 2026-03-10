"""Gemini 1.5 Pro adapter."""

import json
import time
import logging

import google.generativeai as genai

from src.config import get_api_key, CONFIG
from src.models.base import LLMModel, CausalTriple, JudgmentResult, ExtractionError
from src.prompts import format_extraction, format_complexity_judge, format_faithfulness_judge

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


class GeminiModel(LLMModel):
    def __init__(self, model_name: str | None = None):
        api_key = get_api_key("gemini")
        genai.configure(api_key=api_key)
        self.model_name = model_name or CONFIG["models"]["gemini"]
        self._model = genai.GenerativeModel(self.model_name)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Gemini API with retry / exponential backoff. Returns raw text."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                return response.text
            except Exception as exc:
                logger.warning("Gemini attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_BACKOFF_BASE ** attempt)
        raise ExtractionError(f"Gemini failed after {_MAX_RETRIES} attempts.")

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip markdown code fences if the model adds them
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)

    # ── public interface ─────────────────────────────────────────────────────

    def extract(self, passage: str) -> list[CausalTriple]:
        system, user = format_extraction(passage)
        raw = self._call(system, user)
        try:
            data = self._parse_json(raw)
            if not isinstance(data, list):
                data = [data]
            triples = []
            for item in data:
                triples.append(CausalTriple(
                    cause=item.get("cause", ""),
                    connector=item.get("connector", ""),
                    effect=item.get("effect", ""),
                    hedge=item.get("hedge", ""),
                    direction=item.get("direction", "ambiguous"),
                    raw_response=raw,
                ))
            return triples
        except Exception as exc:
            logger.error("Failed to parse extraction response: %s\nRaw: %s", exc, raw)
            return []

    def judge(self, passage: str, triple: CausalTriple) -> JudgmentResult:
        # Complexity
        sys_c, usr_c = format_complexity_judge(passage)
        raw_c = self._call(sys_c, usr_c)
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
        raw_f = self._call(sys_f, usr_f)
        try:
            faith = self._parse_json(raw_f)
        except Exception as exc:
            logger.error("Failed to parse faithfulness response: %s\nRaw: %s", exc, raw_f)
            faith = {"faithful": -1, "faithful_rationale": "parse error", "failure_mode": ""}

        return JudgmentResult(
            complexity_score=int(comp.get("complexity_score", 0)),
            complexity_rationale=comp.get("complexity_rationale", ""),
            faithful=int(faith.get("faithful", -1)),
            faithful_rationale=faith.get("faithful_rationale", ""),
            failure_mode=faith.get("failure_mode", ""),
        )
