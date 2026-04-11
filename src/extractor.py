"""
Extraction orchestration using Gemini Batch API.

Reads outputs/passages.csv, submits batch jobs to extract causal triples,
and writes outputs/extractions.csv.

Usage:
    python -m src.extractor
    python -m src.extractor --held-out-only     # test on held-out set first
    python -m src.extractor --poll              # resume polling in-progress batch jobs
    python -m src.extractor --skip-extracted    # skip already-extracted passages
"""

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import pandas as pd
from tqdm import tqdm
from google.genai import types

from src.config import CONFIG, ROOT
from src.models import get_model
from src.batch_state import BatchStateManager, BatchJobState

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
HELD_OUT_CSV = ROOT / "outputs" / "held_out_passages.csv"
FAILED_PASSAGES_CSV = ROOT / "outputs" / "failed_passages.csv"
HELD_OUT_SIZE: int = CONFIG.get("held_out_size", 20)


# ── Recovery utilities ────────────────────────────────────────────────────────

def recover_partial_json(raw_json: str) -> list[dict]:
    """
    Extract causal triple data from truncated JSON responses.
    
    Extracts whatever fields are present using regex, including incomplete triples.
    Returns triples with empty strings for missing fields.
    
    Args:
        raw_json: Truncated JSON string from LLM response
    
    Returns:
        List of recovered triple dictionaries (with whatever fields were found)
    """
    recovered_triples = []
    
    # Find all triple-like objects - look for blocks with curly braces
    # that contain at least one field
    triple_blocks = re.findall(r'\{[^{}]*?"(?:cause|connector|effect|hedge|direction)"[^{}]*?\}', raw_json, re.DOTALL)
    
    for block in triple_blocks:
        # Extract each field individually - use empty string if not found
        cause_match = re.search(r'"cause"\s*:\s*"([^"]*)"', block)
        connector_match = re.search(r'"connector"\s*:\s*"([^"]*)"', block)
        effect_match = re.search(r'"effect"\s*:\s*"([^"]*)"', block)
        hedge_match = re.search(r'"hedge"\s*:\s*"([^"]*)"', block)
        direction_match = re.search(r'"direction"\s*:\s*"(positive|negative|ambiguous)"', block)
        
        # Build triple with whatever we found (empty string for missing)
        triple = {
            "cause": cause_match.group(1) if cause_match else "",
            "connector": connector_match.group(1) if connector_match else "",
            "effect": effect_match.group(1) if effect_match else "",
            "hedge": hedge_match.group(1) if hedge_match else "",
            "direction": direction_match.group(1) if direction_match else "",
        }
        
        # Include if we found at least one field
        if any(triple.values()):
            recovered_triples.append(triple)
    
    if recovered_triples:
        logger.info("RECOVERY: Extracted %d partial triples from truncated JSON", len(recovered_triples))
    else:
        logger.warning("RECOVERY: No recoverable data found in truncated JSON")
    
    return recovered_triples


# ── Held-out set management ───────────────────────────────────────────────────

def get_held_out(passages_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Return the held-out refinement set.

    If outputs/held_out_passages.csv already exists, load it (so the same 20
    passages are used across runs). Otherwise sample HELD_OUT_SIZE rows
    uniformly and save the file.
    """
    if HELD_OUT_CSV.exists():
        return pd.read_csv(HELD_OUT_CSV)

    sample_df = passages_df[passages_df['period'] != 'great_moderation']
    sample = sample_df.sample(n=min(HELD_OUT_SIZE, len(sample_df)), random_state=random_state)
    sample.to_csv(HELD_OUT_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Held-out set (%d passages) saved to %s", len(sample), HELD_OUT_CSV)
    return sample


def sample_passages_by_period(
    passages_df: pd.DataFrame,
    max_per_period: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample up to max_per_period passages from each economic period.

    For each period, samples min(max_per_period, period_size) passages randomly.
    Uses a fixed random seed for reproducibility across runs.

    Args:
        passages_df: DataFrame with all passages (must have 'period' column)
        max_per_period: Maximum number of passages to sample per period
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        DataFrame with sampled passages concatenated across all periods
    """
    sampled_dfs = []
    for period_name, group in passages_df.groupby('period'):
        sample_size = min(max_per_period, len(group))
        sampled = group.sample(n=sample_size, random_state=random_state)
        sampled_dfs.append(sampled)
        logger.info(
            "Period '%s': sampled %d/%d passages",
            period_name, sample_size, len(group)
        )
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    return result


# ── Extraction loop (legacy sequential version) ──────────────────────────────

def run_extraction(
    passages_df: pd.DataFrame,
    provider: str | None = None,
    skip_extracted: bool = False,
) -> pd.DataFrame:
    """
    Sequential extraction fallback for non-Gemini providers.
    
    Use run_extraction_batch_api() for Gemini (50% cost savings).
    This sequential mode processes one passage at a time.
    """
    model = get_model(provider)

    # Load existing results if skipping already-extracted passages
    existing_ids: set[str] = set()
    existing_records: list[dict] = []
    if skip_extracted and EXTRACTIONS_CSV.exists():
        existing_df = pd.read_csv(EXTRACTIONS_CSV)
        existing_ids = set(existing_df["passage_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info("Resuming — skipping %d already-processed passage IDs", len(existing_ids))

    todo = passages_df[~passages_df["passage_id"].astype(str).isin(existing_ids)]
    logger.info("Processing %d passages (SEQUENTIAL MODE)…", len(todo))

    new_records: list[dict] = []
    triple_counter = 0
    save_interval = 10  # Save every 10 passages

    for idx, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="Extracting"), 1):
        passage_id = str(row["passage_id"])
        passage_text = str(row["text"])

        extract_start = time.time()
        triples = model.extract(passage_text)
        extract_time = time.time() - extract_start
        
        if not triples:
            # Record a null extraction so we know the passage was processed
            new_records.append({
                "triple_id": f"{passage_id}",
                "passage_id": passage_id,
                "meeting_date": row.get("meeting_date", ""),
                "period": row.get("period", ""),
                "text": passage_text,
                "cause": "",
                "connector": "",
                "effect": "",
                "hedge": "",
                "direction": "",
                "raw_response": "",
                "extraction_error": "",  # Empty result is not an error
            })
        else:
            for i, triple in enumerate(triples):
                new_records.append({
                    "triple_id": f"{passage_id}_t{i:03d}",
                    "passage_id": passage_id,
                    "meeting_date": row.get("meeting_date", ""),
                    "period": row.get("period", ""),
                    "text": passage_text,
                    "cause": triple.cause,
                    "connector": triple.connector,
                    "effect": triple.effect,
                    "hedge": triple.hedge,
                    "direction": triple.direction,
                    "raw_response": triple.raw_response,
                    "extraction_error": "",
                })
                triple_counter += 1
        
        # Save progress every save_interval passages
        if idx % save_interval == 0:
            all_records = existing_records + new_records
            result = pd.DataFrame(all_records)
            EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(EXTRACTIONS_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
            logger.debug("Saved progress: %d passages processed", idx)

    # Final save
    all_records = existing_records + new_records
    result = pd.DataFrame(all_records)
    EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(EXTRACTIONS_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(
        "Wrote %d triples (%d passages) to %s",
        triple_counter, len(todo), EXTRACTIONS_CSV,
    )
    return result


# ── Extraction via Batch API ─────────────────────────────────────────────────

def run_extraction_batch_api(
    passages_df: pd.DataFrame,
    provider: str | None = None,
    poll: bool = False,
    skip_extracted: bool = False,
) -> pd.DataFrame:
    """
    Extract causal triples using Gemini Batch API (asynchronous processing).
    
    This replaces the concurrent approach with Gemini's official Batch API:
    1. Build batch requests from passages
    2. Submit batch job to Gemini
    3. Poll for completion (may take hours)
    4. Retrieve and parse results
    
    Args:
        passages_df: DataFrame with passages to extract from
        provider: LLM provider (must be "gemini" for batch API)
        poll: Resume polling in-progress batch jobs
        skip_extracted: Skip passages already in extractions.csv
    
    Returns:
        DataFrame with one row per extracted triple
    """
    model = get_model(provider)
    
    # Batch API only supported for Gemini
    if not hasattr(model, 'submit_batch'):
        raise NotImplementedError(
            f"Batch API not supported for provider: {provider}. "
            "Only Gemini supports batch processing."
        )
    
    state_manager = BatchStateManager()
    
    # Check for existing in-progress batch jobs
    pending_jobs = state_manager.get_pending(job_type="extraction")
    completed_jobs = state_manager.get_completed(job_type="extraction")
    
    # If --poll flag is used, check for both pending AND completed jobs
    if poll:
        if not pending_jobs and not completed_jobs:
            print("\n⚠️  No in-progress or completed extraction batch jobs found.")
            print("   Run without --poll to start a new extraction batch.")
            return pd.DataFrame()
        
        # If there are completed jobs, retrieve them
        if completed_jobs:
            job = completed_jobs[0]
            print(f"\n✅ Found completed batch job:")
            print(f"  - {job.display_name} ({job.job_name})")
            print(f"    State: {job.state}, Created: {job.created_at}")
            print(f"\n📥 Retrieving results...")
            
            # Load passages data for merging
            if not PASSAGES_CSV.exists():
                print(f"⚠️  Warning: {PASSAGES_CSV} not found. Extraction will be missing metadata.")
                passages_df = pd.DataFrame()
            else:
                passages_df = pd.read_csv(PASSAGES_CSV)
            
            # Load existing extractions (if any)
            existing_records = []
            if EXTRACTIONS_CSV.exists():
                existing_df = pd.read_csv(EXTRACTIONS_CSV)
                existing_records = existing_df.to_dict('records')
                print(f"   Found {len(existing_records)} existing extraction records")
            
            # Get batch job and retrieve results
            batch_job = model.get_batch_status(job.job_name)
            inlined_responses = model.retrieve_batch_results(batch_job)
            print(f"✅ Retrieved {len(inlined_responses)} responses")
            
            # Parse results
            print(f"\n🔍 Parsing extraction results...")
            new_records = parse_batch_extraction_results(inlined_responses, model.model_name)
            
            # Merge with passage metadata
            if not passages_df.empty:
                print(f"🔗 Merging with passage metadata...")
                new_records = merge_extraction_with_passages(new_records, passages_df)
            
            # Combine and save
            all_records = existing_records + new_records
            result_df = pd.DataFrame(all_records)
            result_df.to_csv(EXTRACTIONS_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
            
            # Save failed passage IDs for re-extraction
            save_failed_passages(all_records)
            
            print(f"\n✅ Extraction complete!")
            print(f"   Total triples: {len(new_records)}")
            print(f"   Output: {EXTRACTIONS_CSV}")
            
            # Delete the completed job from state
            state_manager.delete(job.job_name)
            print(f"\n🗑️  Removed completed job from batch state")
            
            return result_df
        
        # Otherwise poll pending jobs
        if pending_jobs:
            print(f"\n🔄 Resuming polling for existing batch job:")
            print(f"  - {pending_jobs[0].display_name} ({pending_jobs[0].job_name})")
            print(f"    State: {pending_jobs[0].state}, Created: {pending_jobs[0].created_at}")
            return poll_and_retrieve_batch(pending_jobs[0], model, state_manager)
    
    # If creating a new batch, warn about existing jobs
    if pending_jobs or completed_jobs:
        if pending_jobs:
            print(f"\n⚠️  Found {len(pending_jobs)} in-progress extraction batch job(s):")
            for job in pending_jobs:
                print(f"  - {job.display_name} ({job.job_name})")
                print(f"    State: {job.state}, Created: {job.created_at}")
        
        if completed_jobs:
            print(f"\n⚠️  Found {len(completed_jobs)} completed extraction batch job(s) ready to retrieve:")
            for job in completed_jobs:
                print(f"  - {job.display_name} ({job.job_name})")
                print(f"    State: {job.state}, Created: {job.created_at}")
        
        if pending_jobs:
            response = input("\nContinue polling existing batch? [Y/n]: ").strip().lower()
            if response in ('', 'y', 'yes'):
                # Poll the first pending job
                return poll_and_retrieve_batch(pending_jobs[0], model, state_manager)
            else:
                print("Skipping existing batch. Starting new extraction...")
        elif completed_jobs:
            print("\n💡 Tip: Run with --poll to automatically retrieve completed batch results")
            response = input("Skip completed batches and start new extraction? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("Please run with --poll to retrieve completed batches first.")
                return pd.DataFrame()
    
    # Load existing results if skipping already-extracted passages
    existing_ids: set[str] = set()
    existing_records: list[dict] = []
    if skip_extracted and EXTRACTIONS_CSV.exists():
        existing_df = pd.read_csv(EXTRACTIONS_CSV)
        existing_ids = set(existing_df["passage_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info("Skipping %d already-processed passage IDs", len(existing_ids))
    
    todo = passages_df[~passages_df["passage_id"].astype(str).isin(existing_ids)]
    logger.info("Preparing batch job for %d passages...", len(todo))
    
    if len(todo) == 0:
        logger.info("No new passages to extract. All passages already processed.")
        return pd.DataFrame(existing_records) if existing_records else pd.DataFrame()
    
    # Build batch requests
    print(f"\n📦 Building batch requests for {len(todo)} passages...")
    requests = []
    seen_passage_ids = set()
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Building requests"):
        passage_id = str(row["passage_id"])
        passage_text = str(row["text"])
        
        # Detect duplicate passage IDs in request building
        if passage_id in seen_passage_ids:
            logger.warning("DUPLICATE passage_id detected in batch building: %s", passage_id)
            continue
        seen_passage_ids.add(passage_id)
        
        request = model.build_extraction_request(passage_text, passage_id)
        requests.append(request)
    
    # Verify no duplicates
    if len(requests) != len(seen_passage_ids):
        raise ValueError(f"Request count mismatch: {len(requests)} requests but {len(seen_passage_ids)} unique IDs")
    
    # Submit batch job
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"extraction_{len(todo)}_passages_{timestamp}"
    
    print(f"\n🚀 Submitting batch job: {display_name}")
    print(f"   Requests to submit: {len(requests)}")
    print(f"   Unique passage IDs: {len(seen_passage_ids)}")
    
    batch_job = model.submit_batch(requests, display_name=display_name)
    
    # Save batch state
    job_state = BatchJobState(
        job_name=batch_job.name,
        display_name=display_name,
        job_type="extraction",
        model=model.model_name,
        created_at=datetime.now().isoformat(),
        state=str(batch_job.state),
        total_requests=len(requests),
    )
    state_manager.save(job_state)
    
    print(f"✅ Batch job submitted successfully!")
    print(f"   Job ID: {batch_job.name}")
    print(f"   State: {batch_job.state}")
    print(f"\n⏳ Polling for results (this may take hours)...\n")
    
    # Poll and retrieve results
    return poll_and_retrieve_batch(job_state, model, state_manager, existing_records)


def poll_and_retrieve_batch(
    job_state: BatchJobState,
    model,
    state_manager: BatchStateManager,
    existing_records: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Poll a batch job until completion and retrieve results.
    
    Args:
        job_state: Batch job state to poll
        model: LLM model instance (must have get_batch_status method)
        state_manager: Batch state manager for persistence
        existing_records: Existing extraction records (for resume)
    
    Returns:
        DataFrame with extraction results
    """
    if existing_records is None:
        existing_records = []
    
    # Get polling configuration
    batch_config = CONFIG.get("batch_api", {})
    initial_delay = batch_config.get("poll_initial_delay", 60)
    max_interval = batch_config.get("poll_max_interval", 600)
    backoff_factor = batch_config.get("poll_backoff_factor", 1.5)
    
    current_interval = initial_delay
    start_time = time.time()
    poll_count = 0
    
    try:
        while True:
            poll_count += 1
            elapsed = time.time() - start_time
            
            # Wait before polling
            if poll_count > 1:
                print(f"⏰ Waiting {current_interval:.0f}s before next poll...")
                time.sleep(current_interval)
                current_interval = min(current_interval * backoff_factor, max_interval)
            
            # Poll status
            try:
                batch_job = model.get_batch_status(job_state.job_name)
            except Exception as exc:
                # Check if it's an authentication error
                error_str = str(exc).lower()
                if 'authentication' in error_str or 'permission' in error_str or '403' in error_str:
                    print(f"\n❌ Authentication Error")
                    print(f"   {exc}")
                else:
                    print(f"\n❌ Error polling batch status: {exc}")
                raise
            
            # Update state
            job_state.state = str(batch_job.state)
            job_state.updated_at = datetime.now().isoformat()
            
            if batch_job.completion_stats:
                job_state.completed_requests = batch_job.completion_stats.succeeded_count or 0
                job_state.failed_requests = batch_job.completion_stats.failed_count or 0
            
            state_manager.save(job_state)
            
            # Display status
            elapsed_str = f"{elapsed/60:.1f}min" if elapsed < 3600 else f"{elapsed/3600:.1f}hr"
            print(f"\n📊 Poll #{poll_count} (elapsed: {elapsed_str})")
            print(f"   State: {batch_job.state}")
            
            if batch_job.completion_stats:
                completed = batch_job.completion_stats.succeeded_count or 0
                failed = batch_job.completion_stats.failed_count or 0
                total = job_state.total_requests
                print(f"   Progress: {completed}/{total} completed, {failed} failed")
            
            # Check if done
            if batch_job.done:
                print(f"\n✅ Batch job completed!")
                break
            
            if batch_job.error:
                print(f"\n❌ Batch job failed: {batch_job.error}")
                raise Exception(f"Batch job failed: {batch_job.error}")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Polling interrupted by user.")
        print(f"   Job ID: {job_state.job_name}")
        print(f"   Resume polling later with --poll flag")
        raise
    
    # Retrieve results
    print(f"\n📥 Retrieving batch results...")
    inlined_responses = model.retrieve_batch_results(batch_job)
    
    print(f"✅ Retrieved {len(inlined_responses)} responses")
    
    # Parse results into extraction records
    print(f"\n🔍 Parsing extraction results...")
    new_records = parse_batch_extraction_results(inlined_responses, model.model_name)
    
    # Merge with passage metadata
    if PASSAGES_CSV.exists():
        print(f"🔗 Merging with passage metadata...")
        passages_df = pd.read_csv(PASSAGES_CSV)
        new_records = merge_extraction_with_passages(new_records, passages_df)
    else:
        logger.warning("Passages CSV not found. Extraction will be missing metadata.")
    
    # Combine with existing records
    all_records = existing_records + new_records
    result_df = pd.DataFrame(all_records)
    
    # Save to CSV
    result_df.to_csv(EXTRACTIONS_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Wrote %d triples (%d passages) to %s",
               len(new_records), len(inlined_responses), EXTRACTIONS_CSV)
    
    # Save failed passage IDs for re-extraction
    save_failed_passages(all_records)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Total triples: {len(all_records)}")
    print(f"   New triples: {len(new_records)}")
    print(f"   Output: {EXTRACTIONS_CSV}")
    
    return result_df



def parse_batch_extraction_results(
    inlined_responses: list[types.InlinedResponse],
    model_name: str,
) -> list[dict]:
    """
    Parse batch extraction responses into extraction records.
    
    Args:
        inlined_responses: List of InlinedResponse from batch job
        model_name: Name of the model used
    
    Returns:
        List of extraction record dictionaries
    """
    from src.models.schemas import ExtractionResponseSchema
    from pydantic import ValidationError
    import json
    
    records = []
    
    for response_obj in tqdm(inlined_responses, desc="Parsing responses"):
        # Get metadata
        metadata = response_obj.metadata or {}
        passage_id = metadata.get("passage_id", "unknown")
        
        # Handle batch API errors
        if response_obj.error:
            logger.error("Batch request failed for passage %s: %s", 
                        passage_id, response_obj.error)
            records.append({
                "triple_id": f"{passage_id}_error",
                "passage_id": passage_id,
                "meeting_date": None,
                "period": None,
                "text": None,
                "cause": "",
                "connector": "",
                "effect": "",
                "hedge": "",
                "direction": "",
                "raw_response": str(response_obj.error),
                "extraction_error": f"batch_error: {response_obj.error}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model": model_name,
                "malformed_count": 0,
            })
            continue
        
        # Handle empty responses
        response = response_obj.response
        if not response or not response.text:
            logger.warning("Empty response for passage %s", passage_id)
            records.append({
                "triple_id": f"{passage_id}_empty",
                "passage_id": passage_id,
                "meeting_date": None,
                "period": None,
                "text": None,
                "cause": "",
                "connector": "",
                "effect": "",
                "hedge": "",
                "direction": "",
                "raw_response": "",
                "extraction_error": "empty_response",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model": model_name,
                "malformed_count": 0,
            })
            continue
        
        # Get token usage
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        # Try to parse and validate response
        try:
            extraction_response = ExtractionResponseSchema.model_validate_json(response.text)
            
            # Handle empty triples (normal outcome)
            if not extraction_response.triples:
                records.append({
                    "triple_id": f"{passage_id}_empty",
                    "passage_id": passage_id,
                    "meeting_date": None,
                    "period": None,
                    "text": None,
                    "cause": "",
                    "connector": "",
                    "effect": "",
                    "hedge": "",
                    "direction": "",
                    "raw_response": response.text,
                    "extraction_error": "",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model": model_name,
                    "malformed_count": 0,
                })
            else:
                # Create record for each triple
                for idx, triple_schema in enumerate(extraction_response.triples):
                    records.append({
                        "triple_id": f"{passage_id}_{idx}",
                        "passage_id": passage_id,
                        "meeting_date": None,  # Filled by merge
                        "period": None,  # Filled by merge
                        "text": None,  # Filled by merge
                        "cause": triple_schema.cause,
                        "connector": triple_schema.connector,
                        "effect": triple_schema.effect,
                        "hedge": triple_schema.hedge,
                        "direction": triple_schema.direction,
                        "raw_response": response.text,
                        "extraction_error": "",
                        "prompt_tokens": prompt_tokens if idx == 0 else 0,  # Count once
                        "completion_tokens": completion_tokens if idx == 0 else 0,
                        "model": model_name,
                        "malformed_count": 0,
                    })
        
        except ValidationError as exc:
            # Validation failed - attempt recovery from truncated JSON
            logger.error("Validation failed for passage %s: %s", passage_id, exc)
            
            # Try to recover partial data
            recovered_triples = recover_partial_json(response.text)
            
            if recovered_triples:
                # Create records for recovered triples
                for idx, triple in enumerate(recovered_triples):
                    records.append({
                        "triple_id": f"{passage_id}_recovered_{idx}",
                        "passage_id": passage_id,
                        "meeting_date": None,
                        "period": None,
                        "text": None,
                        "cause": triple["cause"],
                        "connector": triple["connector"],
                        "effect": triple["effect"],
                        "hedge": triple["hedge"],
                        "direction": triple["direction"],
                        "raw_response": response.text,
                        "extraction_error": f"recovered: {str(exc)[:150]}",
                        "prompt_tokens": prompt_tokens if idx == 0 else 0,
                        "completion_tokens": completion_tokens if idx == 0 else 0,
                        "model": model_name,
                    })
            else:
                # No recovery possible - create error record
                records.append({
                    "triple_id": f"{passage_id}_validation_error",
                    "passage_id": passage_id,
                    "meeting_date": None,
                    "period": None,
                    "text": None,
                    "cause": "",
                    "connector": "",
                    "effect": "",
                    "hedge": "",
                    "direction": "",
                    "raw_response": response.text,
                    "extraction_error": f"validation_error: {str(exc)[:200]}",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model": model_name,
                })
        
        except Exception as exc:
            # JSON parsing or other error
            logger.error("Parse error for passage %s: %s", passage_id, exc)
            records.append({
                "triple_id": f"{passage_id}_parse_error",
                "passage_id": passage_id,
                "meeting_date": None,
                "period": None,
                "text": None,
                "cause": "",
                "connector": "",
                "effect": "",
                "hedge": "",
                "direction": "",
                "raw_response": response.text,
                "extraction_error": f"parse_error: {str(exc)[:200]}",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model_name,
                "malformed_count": 0,
            })
    
    return records




def merge_extraction_with_passages(
    extraction_records: list[dict],
    passages_df: pd.DataFrame,
) -> list[dict]:
    """
    Merge extraction records with passage metadata (meeting_date, period, text).
    
    Args:
        extraction_records: List of extraction record dicts (with passage_id)
        passages_df: DataFrame with columns: passage_id, meeting_date, period, text
    
    Returns:
        List of extraction records with merged metadata
    """
    # Create a lookup dict for fast access
    passage_lookup = {}
    for _, row in passages_df.iterrows():
        passage_id = str(row["passage_id"])
        passage_lookup[passage_id] = {
            "meeting_date": row.get("meeting_date"),
            "period": row.get("period"),
            "text": row.get("text"),
        }
    
    # Merge metadata into each record
    merged_records = []
    for record in extraction_records:
        passage_id = str(record.get("passage_id", ""))
        passage_data = passage_lookup.get(passage_id, {})
        
        # Update record with passage metadata
        record["meeting_date"] = passage_data.get("meeting_date")
        record["period"] = passage_data.get("period")
        record["text"] = passage_data.get("text")
        
        merged_records.append(record)
    
    return merged_records


# ── Single-passage extraction for notebook testing ───────────────────────────

def extract_single_passage(
    passage_text: str,
    provider: str | None = None,
    custom_prompt: str | None = None,
    passage_metadata: dict | None = None,
) -> pd.DataFrame:
    """
    Extract causal triples from a single passage (for notebook experimentation).
    
    This utility function allows you to test different extraction prompts on
    individual passages without running the full pipeline. Useful for:
    - Prompt development and comparison
    - Quick extraction testing
    - Debugging specific passages
    
    Args:
        passage_text: The text to extract causal triples from
        provider: LLM provider override (e.g., "gemini", "openai", "github").
                 If None, uses the provider from config.yaml
        custom_prompt: Optional custom user prompt string. Must contain {passage}
                      placeholder. If None, uses the default EXTRACTION_USER.
                      Example: custom_prompt=EXTRACTION_USER_OLD
        passage_metadata: Optional dict with keys: passage_id, meeting_date, period.
                         Used to populate metadata columns in the output DataFrame.
    
    Returns:
        DataFrame with same schema as extractions.csv:
        - triple_id, passage_id, meeting_date, period, text
        - cause, connector, effect, hedge, direction
        - raw_response, extraction_error, extract_time_s
        
        Returns one row per extracted triple. If no triples found, returns
        one row with empty cause/effect/etc and extraction_error="" (empty string).
    
    Example:
        >>> from src.prompts import EXTRACTION_USER_OLD
        >>> from src.extractor import extract_single_passage
        >>> 
        >>> passage = "Rising oil prices boosted inflation expectations."
        >>> 
        >>> # Test with default prompt
        >>> df1 = extract_single_passage(passage)
        >>> 
        >>> # Test with old prompt
        >>> df2 = extract_single_passage(passage, custom_prompt=EXTRACTION_USER_OLD)
        >>> 
        >>> # Compare results
        >>> print(df1[["cause", "connector", "effect"]])
        >>> print(df2[["cause", "connector", "effect"]])
    """
    # Get model
    model = get_model(provider)
    
    # Prepare prompts
    if custom_prompt is not None:
        # User provided custom prompt - format it with the passage
        system_prompt = EXTRACTION_SYSTEM
        user_prompt = custom_prompt.format(passage=passage_text)
    else:
        # Use default prompts
        system_prompt, user_prompt = format_extraction(passage_text)
    
    # Extract with timing
    extract_start = time.time()
    try:
        # Call the model's internal _call method to get raw JSON
        raw_json = model._call(system_prompt, user_prompt)
        
        # Parse JSON into list of dicts
        data = model._parse_json(raw_json)
        if not isinstance(data, list):
            data = [data] if data else []
        
        # Convert to CausalTriple objects
        from src.models.base import CausalTriple
        triples = []
        for item in data:
            triples.append(CausalTriple(
                cause=item.get("cause", ""),
                connector=item.get("connector", ""),
                effect=item.get("effect", ""),
                hedge=item.get("hedge", ""),
                direction=item.get("direction", "ambiguous"),
                raw_response=raw_json,
            ))
        
        extraction_error = ""
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}")
        triples = []
        raw_json = ""
        extraction_error = str(exc)
    
    extract_time = time.time() - extract_start
    
    # Extract metadata with defaults
    metadata = passage_metadata or {}
    passage_id = metadata.get("passage_id", "test_passage")
    meeting_date = metadata.get("meeting_date", "")
    period = metadata.get("period", "")
    
    # Build DataFrame records
    records = []
    if not triples:
        # No triples extracted - create one record with empty fields
        records.append({
            "triple_id": passage_id,
            "passage_id": passage_id,
            "meeting_date": meeting_date,
            "period": period,
            "text": passage_text,
            "cause": "",
            "connector": "",
            "effect": "",
            "hedge": "",
            "direction": "",
            "raw_response": raw_json,
            "extraction_error": extraction_error or "",  # Empty result is not an error unless there was an actual error
            "extract_time_s": extract_time,
        })
    else:
        # One record per triple
        for i, triple in enumerate(triples):
            records.append({
                "triple_id": f"{passage_id}_t{i:03d}",
                "passage_id": passage_id,
                "meeting_date": meeting_date,
                "period": period,
                "text": passage_text,
                "cause": triple.cause,
                "connector": triple.connector,
                "effect": triple.effect,
                "hedge": triple.hedge,
                "direction": triple.direction,
                "raw_response": triple.raw_response,
                "extraction_error": extraction_error,
                "extract_time_s": extract_time,
            })
    
    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run causal triple extraction.")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini or openai")
    parser.add_argument(
        "--held-out-only", action="store_true",
        help="Run extraction only on the 20-passage held-out refinement set.",
    )
    parser.add_argument(
        "--poll", action="store_true",
        help="Resume polling in-progress batch jobs (retrieves results if complete).",
    )
    parser.add_argument(
        "--skip-extracted", action="store_true",
        help="Skip passages already in extractions.csv and only extract remaining passages.",
    )
    args = parser.parse_args()

    if not PASSAGES_CSV.exists():
        print(f"ERROR: {PASSAGES_CSV} not found. Run src/preprocessor.py first.")
        return

    passages_df = pd.read_csv(PASSAGES_CSV)

    # Optional passage limit: sample max N passages per period
    max_per_period = CONFIG.get("passage", {}).get("max_passages_per_period")
    if max_per_period is not None:
        logger.info("Sampling passages: max %d per period...", max_per_period)
        passages_df = sample_passages_by_period(passages_df, max_per_period)
        logger.info("Total passages after sampling: %d", len(passages_df))

    # Optional passage balancing: subsample each period to min(period_counts)
    if CONFIG.get("passage", {}).get("balance_passages", False):
        min_count = passages_df.groupby("period")["passage_id"].count().min()
        passages_df = (
            passages_df
            .groupby("period", group_keys=False)
            .apply(lambda g: g.sample(n=min_count, random_state=42))
            .reset_index(drop=True)
        )
        logger.info("Balanced passages to %d per period (%d total).", min_count, len(passages_df))

    if args.held_out_only:
        passages_df = get_held_out(passages_df)
        logger.info("Running on held-out set (%d passages).", len(passages_df))

    # Run extraction via Batch API
    logger.info("Running extraction via Gemini Batch API (asynchronous processing)")
    run_extraction_batch_api(passages_df, provider=args.provider, poll=args.poll, skip_extracted=args.skip_extracted)


if __name__ == "__main__":
    main()
