"""Gemini adapter using the google-genai SDK (replaces deprecated google-generativeai)."""

import asyncio
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.config import get_api_key, CONFIG
from src.models.base import LLMModel, CausalTriple, JudgmentResult, ExtractionError
from src.models.schemas import ExtractionResponseSchema, ComplexityJudgmentSchema, FaithfulnessJudgmentSchema
from src.prompts import format_extraction, format_complexity_judge, format_faithfulness_judge

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


class GeminiModel(LLMModel):
    def __init__(self, model_name: str | None = None):
        api_key = get_api_key("gemini")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name or CONFIG["models"]["gemini"]
        self._executor = ThreadPoolExecutor(max_workers=20)  # Pool for async operations

    # ── internal helpers ─────────────────────────────────────────────────────

    def _call(self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel] | None = None) -> str:
        """Call the Gemini API with retry / exponential backoff. Returns raw text.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User prompt/question
            response_schema: Optional Pydantic BaseModel class for structured output
        
        Returns:
            Raw JSON text from the model
        """
        start_time = time.time()
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                attempt_start = time.time()
                
                # Build config with optional schema
                config_params = {
                    "system_instruction": system_prompt,
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                }
                
                if response_schema is not None:
                    config_params["response_json_schema"] = response_schema.model_json_schema()
                
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**config_params),
                )
                
                attempt_time = time.time() - attempt_start
                total_time = time.time() - start_time
                logger.debug("Gemini API call succeeded (attempt %d, %.2fs this attempt, %.2fs total)", 
                            attempt, attempt_time, total_time)
                return response.text
            except Exception as exc:
                attempt_time = time.time() - attempt_start
                logger.warning("Gemini attempt %d/%d failed after %.2fs: %s", attempt, _MAX_RETRIES, attempt_time, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_BACKOFF_BASE ** attempt)
        raise ExtractionError(f"Gemini failed after {_MAX_RETRIES} attempts.")

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)

    # ── public interface ─────────────────────────────────────────────────────

    def extract(self, passage: str) -> list[CausalTriple]:
        start_time = time.time()
        passage_len = len(passage)
        
        # Format prompts
        prompt_start = time.time()
        system, user = format_extraction(passage)
        prompt_time = time.time() - prompt_start
        
        # API call with schema
        api_start = time.time()
        raw = self._call(system, user, response_schema=ExtractionResponseSchema)
        api_time = time.time() - api_start
        
        # Parse and validate with Pydantic
        parse_start = time.time()
        try:
            # Validate JSON against schema
            extraction_response = ExtractionResponseSchema.model_validate_json(raw)
            
            # Convert Pydantic models to CausalTriple dataclass instances
            triples = []
            for triple_schema in extraction_response.triples:
                triples.append(CausalTriple(
                    cause=triple_schema.cause,
                    connector=triple_schema.connector,
                    effect=triple_schema.effect,
                    hedge=triple_schema.hedge,
                    direction=triple_schema.direction,
                    strength=triple_schema.strength,
                    type=triple_schema.type,
                    raw_response=raw,
                ))
            
            parse_time = time.time() - parse_start
            total_time = time.time() - start_time
            logger.info(
                "Extraction complete: %d triples, passage_len=%d | "
                "prompt=%.2fs, api=%.2fs, parse=%.2fs, total=%.2fs",
                len(triples), passage_len, prompt_time, api_time, parse_time, total_time
            )
            return triples
            
        except ValidationError as exc:
            parse_time = time.time() - parse_start
            total_time = time.time() - start_time
            logger.error("Pydantic validation failed after %.2fs: %s\nRaw: %s", 
                        parse_time, exc, raw)
            return []
        except Exception as exc:
            parse_time = time.time() - parse_start
            total_time = time.time() - start_time
            logger.error("Failed to parse extraction response after %.2fs: %s\nRaw: %s", 
                        parse_time, exc, raw)
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
        # Complexity judgment with schema
        sys_c, usr_c = format_complexity_judge(passage)
        raw_c = self._call(sys_c, usr_c, response_schema=ComplexityJudgmentSchema)
        
        try:
            complexity_judgment = ComplexityJudgmentSchema.model_validate_json(raw_c)
            complexity_score = complexity_judgment.complexity_score
            complexity_rationale = complexity_judgment.complexity_rationale
        except ValidationError as exc:
            logger.error("Complexity judgment validation failed: %s\nRaw: %s", exc, raw_c)
            complexity_score = 0
            complexity_rationale = "validation error"
        except Exception as exc:
            logger.error("Failed to parse complexity response: %s\nRaw: %s", exc, raw_c)
            complexity_score = 0
            complexity_rationale = "parse error"

        # Faithfulness judgment with schema
        sys_f, usr_f = format_faithfulness_judge(
            passage, triple.cause, triple.connector,
            triple.effect, triple.hedge, triple.direction,
        )
        raw_f = self._call(sys_f, usr_f, response_schema=FaithfulnessJudgmentSchema)
        
        try:
            faithfulness_judgment = FaithfulnessJudgmentSchema.model_validate_json(raw_f)
            faithful = faithfulness_judgment.faithful
            faithful_rationale = faithfulness_judgment.faithful_rationale
            failure_mode = faithfulness_judgment.failure_mode
        except ValidationError as exc:
            logger.error("Faithfulness judgment validation failed: %s\nRaw: %s", exc, raw_f)
            faithful = -1
            faithful_rationale = "validation error"
            failure_mode = ""
        except Exception as exc:
            logger.error("Failed to parse faithfulness response: %s\nRaw: %s", exc, raw_f)
            faithful = -1
            faithful_rationale = "parse error"
            failure_mode = ""

        return JudgmentResult(
            complexity_score=complexity_score,
            complexity_rationale=complexity_rationale,
            faithful=faithful,
            faithful_rationale=faithful_rationale,
            failure_mode=failure_mode,
        )
