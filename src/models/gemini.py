"""Gemini adapter using the google-genai SDK (replaces deprecated google-generativeai)."""

import json
import time
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.config import get_api_key, CONFIG
from src.models.base import LLMModel, CausalTriple, JudgmentResult, ExtractionError, TokenUsage
from src.models.schemas import ExtractionResponseSchema, CausalTripleSchema, ComplexityJudgmentSchema, FaithfulnessJudgmentSchema
from src.prompts import format_extraction, format_complexity_judge, format_faithfulness_judge

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


class GeminiModel(LLMModel):
    def __init__(self, model_name: str | None = None):
        api_key = get_api_key("gemini")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name or CONFIG["models"]["gemini"]

    # ── internal helpers ─────────────────────────────────────────────────────

    def _call(self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel] | None = None) -> tuple[str, TokenUsage]:
        """Call the Gemini API with retry / exponential backoff. Returns raw text and token usage.
        
        Args:
            system_prompt: System instruction for the model
            user_prompt: User prompt/question
            response_schema: Optional Pydantic BaseModel class for structured output
        
        Returns:
            Tuple of (raw JSON text, TokenUsage object)
        """
        start_time = time.time()
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                attempt_start = time.time()
                
                # Build config with optional schema and thinking tokens
                config_params = {
                    "system_instruction": system_prompt,
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                }
                
                if response_schema is not None:
                    config_params["response_json_schema"] = response_schema.model_json_schema()
                
                # Add max_output_tokens: thinking models use max_thinking_tokens,
                # regular models use max_output_tokens (higher to avoid truncation)
                if "thinking" in self.model_name.lower() and CONFIG.get("max_thinking_tokens"):
                    config_params["max_output_tokens"] = CONFIG["max_thinking_tokens"]
                elif CONFIG.get("max_output_tokens"):
                    config_params["max_output_tokens"] = CONFIG["max_output_tokens"]
                
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**config_params),
                )
                
                # Extract token usage
                usage = TokenUsage()
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage.prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    usage.completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                
                attempt_time = time.time() - attempt_start
                total_time = time.time() - start_time
                logger.debug("Gemini API call succeeded (attempt %d, %.2fs this attempt, %.2fs total, %d input / %d output tokens)", 
                            attempt, attempt_time, total_time, usage.prompt_tokens, usage.completion_tokens)
                return response.text, usage
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
        raw, usage = self._call(system, user, response_schema=ExtractionResponseSchema)
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
                    raw_response=raw,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                ))
            
            parse_time = time.time() - parse_start
            total_time = time.time() - start_time
            logger.info(
                "Extraction complete: %d triples, passage_len=%d, tokens=%d/%d | "
                "prompt=%.2fs, api=%.2fs, parse=%.2fs, total=%.2fs",
                len(triples), passage_len, usage.prompt_tokens, usage.completion_tokens,
                prompt_time, api_time, parse_time, total_time
            )
            return triples
            
        except ValidationError as exc:
            parse_time = time.time() - parse_start
            total_time = time.time() - start_time
            
            # Attempt to recover from truncated JSON (EOF errors)
            # Try to extract complete triple objects from the truncated response
            recovered_triples = []
            try:
                # Find complete triple objects that have all 5 fields
                # Pattern: object with all fields including direction at the end (indicates completeness)
                import re
                pattern = r'\{[^{}]*?"cause"[^{}]*?"connector"[^{}]*?"effect"[^{}]*?"hedge"[^{}]*?"direction"\s*:\s*"(?:positive|negative|ambiguous)"\s*\}'
                matches = re.findall(pattern, raw, re.DOTALL)
                
                for match in matches:
                    try:
                        # Parse the JSON object
                        triple_obj = json.loads(match)
                        # Validate it has all required fields
                        if all(k in triple_obj for k in ['cause', 'connector', 'effect', 'hedge', 'direction']):
                            validated_triple = CausalTripleSchema.model_validate(triple_obj)
                            recovered_triples.append(CausalTriple(
                                cause=validated_triple.cause,
                                connector=validated_triple.connector,
                                effect=validated_triple.effect,
                                hedge=validated_triple.hedge,
                                direction=validated_triple.direction,
                                raw_response=raw,
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                            ))
                    except (json.JSONDecodeError, ValidationError):
                        # Skip invalid objects
                        pass
                
                if recovered_triples:
                    logger.warning(
                        "TRUNCATION RECOVERY: Extracted %d complete triples from truncated JSON. "
                        "Original validation error: %s",
                        len(recovered_triples), str(exc)[:100]
                    )
                    return recovered_triples
                    
            except Exception as recovery_exc:
                logger.debug("Truncation recovery attempt failed: %s", recovery_exc)
            
            # Attempt to recover from array-format responses
            # LLM sometimes returns arrays instead of objects: ["cause", "connector", "effect", "hedge", "direction"]
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "triples" in parsed:
                    recovered_triples = []
                    triples_data = parsed["triples"]
                    
                    if isinstance(triples_data, list):
                        for item in triples_data:
                            # Check if item is an array instead of object
                            if isinstance(item, (list, tuple)) and len(item) == 5:
                                # Map array positions to schema fields
                                triple_dict = {
                                    "cause": item[0],
                                    "connector": item[1],
                                    "effect": item[2],
                                    "hedge": item[3] if item[3] else "",
                                    "direction": item[4]
                                }
                                # Validate the recovered object
                                try:
                                    validated_triple = CausalTripleSchema.model_validate(triple_dict)
                                    recovered_triples.append(CausalTriple(
                                        cause=validated_triple.cause,
                                        connector=validated_triple.connector,
                                        effect=validated_triple.effect,
                                        hedge=validated_triple.hedge,
                                        direction=validated_triple.direction,
                                        raw_response=raw,
                                        prompt_tokens=usage.prompt_tokens,
                                        completion_tokens=usage.completion_tokens,
                                    ))
                                except ValidationError as ve:
                                    logger.warning("Array recovery failed for item: %s - %s", item, ve)
                            elif isinstance(item, dict):
                                # Already an object, try to validate it
                                try:
                                    validated_triple = CausalTripleSchema.model_validate(item)
                                    recovered_triples.append(CausalTriple(
                                        cause=validated_triple.cause,
                                        connector=validated_triple.connector,
                                        effect=validated_triple.effect,
                                        hedge=validated_triple.hedge,
                                        direction=validated_triple.direction,
                                        raw_response=raw,
                                        prompt_tokens=usage.prompt_tokens,
                                        completion_tokens=usage.completion_tokens,
                                    ))
                                except ValidationError:
                                    pass  # Skip invalid objects
                    
                    if recovered_triples:
                        logger.warning("ARRAY RECOVERY: Successfully recovered %d triples from array format. "
                                      "Prompt should prevent this - check if LLM is following schema.", 
                                      len(recovered_triples))
                        return recovered_triples
            except Exception as recovery_exc:
                logger.debug("Array recovery attempt failed: %s", recovery_exc)
            
            # If recovery failed, log original error and return empty
            logger.error("Pydantic validation failed after %.2fs: %s\nRaw: %s", 
                        parse_time, exc, raw)
            return []
        except Exception as exc:
            parse_time = time.time() - parse_start
            total_time = time.time() - start_time
            logger.error("Failed to parse extraction response after %.2fs: %s\nRaw: %s", 
                        parse_time, exc, raw)
            return []


    # ── Batch API methods ────────────────────────────────────────────────────
    
    def submit_batch(self, requests: list[types.InlinedRequest], display_name: str | None = None) -> types.BatchJob:
        """
        Submit a batch job to Gemini Batch API with retry logic.
        
        Args:
            requests: List of InlinedRequest objects to process in batch
            display_name: Optional human-readable name for the batch job
        
        Returns:
            BatchJob object with job ID and status
        """
        # Verify no duplicate passage IDs in metadata
        passage_ids = [req.metadata.get("passage_id") for req in requests if req.metadata]
        unique_ids = set(passage_ids)
        if len(passage_ids) != len(unique_ids):
            duplicates = len(passage_ids) - len(unique_ids)
            logger.error("CRITICAL: Batch has %d duplicate requests! Unique: %d, Total: %d", 
                        duplicates, len(unique_ids), len(passage_ids))
            raise ValueError(f"Batch contains {duplicates} duplicate passage IDs. This would waste tokens!")
        
        logger.info("Submitting batch job with %d requests (display_name: %s)", 
                   len(requests), display_name or "None")
        
        config = None
        if display_name:
            config = types.CreateBatchJobConfig(display_name=display_name)
        
        # Retry logic for network errors
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                batch_job = self._client.batches.create(
                    model=self.model_name,
                    src=requests,
                    config=config,
                )
                
                logger.info("Batch job created: name=%s, state=%s", batch_job.name, batch_job.state)
                return batch_job
                
            except Exception as exc:
                error_msg = str(exc).lower()
                # Check if it's a network/connectivity error
                if any(keyword in error_msg for keyword in ['connect', 'network', 'timeout', 'getaddrinfo']):
                    logger.warning("Network error on batch submission attempt %d/%d: %s", 
                                 attempt, _MAX_RETRIES, exc)
                    if attempt < _MAX_RETRIES:
                        delay = _BACKOFF_BASE ** attempt
                        logger.info("Retrying in %.1f seconds...", delay)
                        time.sleep(delay)
                    else:
                        raise ExtractionError(
                            f"Failed to submit batch job after {_MAX_RETRIES} attempts due to network errors. "
                            f"Please check your internet connection and try again."
                        ) from exc
                else:
                    # Non-network error, raise immediately
                    raise
        
        raise ExtractionError(f"Failed to submit batch job after {_MAX_RETRIES} attempts.")
    
    def get_batch_status(self, batch_name: str) -> types.BatchJob:
        """
        Get the current status of a batch job with retry logic.
        
        Args:
            batch_name: The batch job name/ID returned from submit_batch()
        
        Returns:
            BatchJob object with current status
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._client.batches.get(name=batch_name)
            except Exception as exc:
                error_msg = str(exc).lower()
                
                # Check for authentication/permission errors
                if 'permission_denied' in error_msg or '403' in error_msg or 'unregistered callers' in error_msg:
                    raise ExtractionError(
                        f"Authentication error: API key does not have permission to access Batch API.\n"
                        f"Error: {exc}"
                    )
                
                # Check if it's a network/connectivity error
                if any(keyword in error_msg for keyword in ['connect', 'network', 'timeout', 'getaddrinfo']):
                    logger.warning("Network error polling batch status attempt %d/%d: %s", 
                                 attempt, _MAX_RETRIES, exc)
                    if attempt < _MAX_RETRIES:
                        delay = _BACKOFF_BASE ** attempt
                        logger.info("Retrying in %.1f seconds...", delay)
                        time.sleep(delay)
                    else:
                        logger.error("Failed to poll batch status after %d attempts. Network unreachable.", _MAX_RETRIES)
                        # Return None or re-raise - caller should handle
                        raise
                else:
                    # Non-network error, raise immediately
                    raise
        
        raise ExtractionError(f"Failed to get batch status after {_MAX_RETRIES} attempts.")
    
    def retrieve_batch_results(self, batch_job: types.BatchJob) -> list[types.InlinedResponse]:
        """
        Retrieve results from a completed batch job.
        
        Args:
            batch_job: Completed BatchJob object
        
        Returns:
            List of InlinedResponse objects (one per input request)
        
        Raises:
            ExtractionError: If batch job failed or not completed
        """
        if not batch_job.done:
            raise ExtractionError(f"Batch job {batch_job.name} not completed (state: {batch_job.state})")
        
        if batch_job.error:
            raise ExtractionError(f"Batch job {batch_job.name} failed: {batch_job.error}")
        
        # Get inlined responses (for Gemini Developer API)
        if batch_job.dest and hasattr(batch_job.dest, 'inlined_responses'):
            responses = batch_job.dest.inlined_responses or []
            logger.info("Retrieved %d inlined responses from batch job %s", 
                       len(responses), batch_job.name)
            return responses
        
        # Fallback: GCS/BigQuery (for Vertex AI)
        if batch_job.dest and batch_job.dest.gcs_uri:
            raise NotImplementedError(
                "GCS result retrieval not yet implemented. "
                f"Results available at: {batch_job.dest.gcs_uri}. "
                "This typically requires Vertex AI setup."
            )
        
        if batch_job.dest and batch_job.dest.bigquery_uri:
            raise NotImplementedError(
                "BigQuery result retrieval not yet implemented. "
                f"Results available at: {batch_job.dest.bigquery_uri}"
            )
        
        raise ExtractionError(f"Batch job {batch_job.name} has no accessible results")
    
    def cancel_batch(self, batch_name: str) -> None:
        """Cancel a running batch job."""
        logger.info("Canceling batch job: %s", batch_name)
        self._client.batches.cancel(name=batch_name)
    
    def build_extraction_request(self, passage: str, passage_id: str) -> types.InlinedRequest:
        """
        Build a batch request for causal triple extraction.
        
        Args:
            passage: Text passage to extract from
            passage_id: Unique identifier for metadata tracking
        
        Returns:
            InlinedRequest configured for extraction
        """
        system, user = format_extraction(passage)
        
        config_params = {
            "system_instruction": system,
            "temperature": 0.0,
            "response_mime_type": "application/json",
            # NOTE: response_json_schema causes Gemini to output escaped strings instead of objects
            # Rely on prompt instructions for structure instead
        }
        
        if "thinking" in self.model_name.lower() and CONFIG.get("max_thinking_tokens"):
            config_params["max_output_tokens"] = CONFIG["max_thinking_tokens"]
        elif CONFIG.get("max_output_tokens"):
            config_params["max_output_tokens"] = CONFIG["max_output_tokens"]
        
        return types.InlinedRequest(
            model=self.model_name,
            contents=user,
            config=types.GenerateContentConfig(**config_params),
            metadata={"passage_id": passage_id, "task": "extraction"},
        )
    
    def build_judgment_request(self, passage: str, triple: CausalTriple, 
                               passage_id: str, triple_idx: int, 
                               judgment_type: str) -> types.InlinedRequest:
        """
        Build a batch request for complexity or faithfulness judgment.
        
        Args:
            passage: Original passage text
            triple: CausalTriple to judge
            passage_id: Passage identifier
            triple_idx: Index of triple within passage
            judgment_type: "complexity" or "faithfulness"
        
        Returns:
            InlinedRequest configured for judgment
        """
        if judgment_type == "complexity":
            system, user = format_complexity_judge(passage)
            schema = ComplexityJudgmentSchema.model_json_schema()
        elif judgment_type == "faithfulness":
            system, user = format_faithfulness_judge(
                passage, triple.cause, triple.connector,
                triple.effect, triple.hedge, triple.direction,
            )
            schema = FaithfulnessJudgmentSchema.model_json_schema()
        else:
            raise ValueError(f"Invalid judgment_type: {judgment_type}")
        
        config_params = {
            "system_instruction": system,
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }
        
        if "thinking" in self.model_name.lower() and CONFIG.get("max_thinking_tokens"):
            config_params["max_output_tokens"] = CONFIG["max_thinking_tokens"]
        elif CONFIG.get("max_output_tokens"):
            config_params["max_output_tokens"] = CONFIG["max_output_tokens"]
        
        return types.InlinedRequest(
            model=self.model_name,
            contents=user,
            config=types.GenerateContentConfig(**config_params),
            metadata={
                "passage_id": passage_id,
                "triple_idx": str(triple_idx),
                "task": judgment_type,
            },
        )
    
    # ── Judgment methods ─────────────────────────────────────────────────────
    
    def judge(self, passage: str, triple: CausalTriple) -> JudgmentResult:
        # Complexity judgment with schema
        sys_c, usr_c = format_complexity_judge(passage)
        raw_c, usage_c = self._call(sys_c, usr_c, response_schema=ComplexityJudgmentSchema)
        
        try:
            complexity_judgment = ComplexityJudgmentSchema.model_validate_json(raw_c)
            complexity_score = complexity_judgment.complexity_score
        except ValidationError as exc:
            logger.error("Complexity judgment validation failed: %s\nRaw: %s", exc, raw_c)
            complexity_score = 0
        except Exception as exc:
            logger.error("Failed to parse complexity response: %s\nRaw: %s", exc, raw_c)
            complexity_score = 0

        # Faithfulness judgment with schema
        sys_f, usr_f = format_faithfulness_judge(
            passage, triple.cause, triple.connector,
            triple.effect, triple.hedge, triple.direction,
        )
        raw_f, usage_f = self._call(sys_f, usr_f, response_schema=FaithfulnessJudgmentSchema)
        
        try:
            faithfulness_judgment = FaithfulnessJudgmentSchema.model_validate_json(raw_f)
            faithful = faithfulness_judgment.faithful
            failure_mode = faithfulness_judgment.failure_mode
        except ValidationError as exc:
            logger.error("Faithfulness judgment validation failed: %s\nRaw: %s", exc, raw_f)
            faithful = -1
            failure_mode = ""
        except Exception as exc:
            logger.error("Failed to parse faithfulness response: %s\nRaw: %s", exc, raw_f)
            faithful = -1
            failure_mode = ""

        return JudgmentResult(
            complexity_score=complexity_score,
            faithful=faithful,
            failure_mode=failure_mode,
        )
