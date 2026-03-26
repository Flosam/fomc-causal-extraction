"""
LLM-as-a-judge orchestration using Gemini Batch API.

Reads outputs/passages.csv + outputs/extractions.csv, submits batch jobs
to score triples for complexity (1–5) and faithfulness (0/1), and writes
outputs/judge_scores.csv.

Usage:
    python -m src.judge
    python -m src.judge --poll           # resume polling in-progress batch jobs
    python -m src.judge --skip-judged    # skip already-judged triples
"""

import argparse
import logging
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from google.genai import types

from src.config import ROOT, CONFIG
from src.models import get_model
from src.batch_state import BatchStateManager, BatchJobState

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
JUDGE_SCORES_CSV = ROOT / "outputs" / "judge_scores.csv"


# ── Judgment via Batch API ───────────────────────────────────────────────────

def run_judge_batch_api(
    passages_df: pd.DataFrame,
    extractions_df: pd.DataFrame,
    provider: str | None = None,
    poll: bool = False,
    skip_judged: bool = False,
) -> pd.DataFrame:
    """
    Judge triples using Gemini Batch API (asynchronous processing).
    
    Submits two separate batch jobs (complexity and faithfulness scoring)
    and polls for completion.
    
    Args:
        passages_df: DataFrame with passage texts
        extractions_df: DataFrame with extracted triples
        provider: LLM provider (must be "gemini" for batch API)
        poll: Resume polling in-progress batch jobs
        skip_judged: Skip triples already judged in judge_scores.csv
    
    Returns:
        DataFrame with judge scores
    """
    from src.models.base import CausalTriple
    
    model = get_model(provider)
    
    # Batch API only supported for Gemini
    if not hasattr(model, 'submit_batch'):
        raise NotImplementedError(
            f"Batch API not supported for provider: {provider}. "
            "Only Gemini supports batch processing."
        )
    
    state_manager = BatchStateManager()
    
    # Check for existing in-progress batch jobs
    pending_jobs = state_manager.get_pending(job_type="judgment")
    
    # If --poll flag is used, ONLY poll existing jobs (don't create new ones)
    if poll:
        if not pending_jobs:
            print("\n⚠️  No in-progress judgment batch jobs found.")
            print("   Run without --poll to start a new judgment batch.")
            return pd.DataFrame()
        
        # Poll existing batches
        print(f"\n🔄 Resuming polling for existing batch job(s):")
        for job in pending_jobs:
            print(f"  - {job.display_name} ({job.job_name})")
            print(f"    State: {job.state}, Created: {job.created_at}")
        return poll_and_retrieve_judgment_batches(pending_jobs, model, state_manager, passages_df, extractions_df)
    
    # If creating a new batch, warn about existing jobs
    if pending_jobs:
        print(f"\n⚠️  Found {len(pending_jobs)} in-progress judgment batch job(s):")
        for job in pending_jobs:
            print(f"  - {job.display_name} ({job.job_name})")
            print(f"    State: {job.state}, Created: {job.created_at}")
        
        response = input("\nContinue polling existing batch? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            # Poll existing batches
            return poll_and_retrieve_judgment_batches(pending_jobs, model, state_manager, passages_df, extractions_df)
        else:
            print("Skipping existing batch. Starting new judgment...")
    
    # Index passages by passage_id for quick lookup
    passage_lookup = passages_df.set_index("passage_id")["text"].to_dict()
    
    # Filter out null-extraction rows
    valid = extractions_df[extractions_df["extraction_error"] != "no_triples"].copy()
    valid = valid[valid["cause"].astype(str).str.strip() != ""]
    
    # Load existing results if skipping already-judged triples
    existing_ids: set[str] = set()
    existing_records: list[dict] = []
    if skip_judged and JUDGE_SCORES_CSV.exists():
        existing_df = pd.read_csv(JUDGE_SCORES_CSV)
        existing_ids = set(existing_df["triple_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info("Skipping %d already-judged triple IDs", len(existing_ids))
    
    todo = valid[~valid["triple_id"].astype(str).isin(existing_ids)]
    logger.info("Preparing batch judgment for %d triples...", len(todo))
    
    if len(todo) == 0:
        logger.info("No new triples to judge. All triples already judged.")
        return pd.DataFrame(existing_records) if existing_records else pd.DataFrame()
    
    # Build batch requests (combined complexity + faithfulness in single request)
    print(f"\n📦 Building batch requests for {len(todo)} triples...")
    requests = []
    triple_metadata = []  # Store for result mapping
    
    for idx, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="Building requests")):
        passage_id = str(row["passage_id"])
        triple_id = str(row["triple_id"])
        passage_text = passage_lookup.get(passage_id, "")
        
        triple = CausalTriple(
            cause=str(row["cause"]),
            connector=str(row["connector"]),
            effect=str(row["effect"]),
            hedge=str(row["hedge"]),
            direction=str(row["direction"]),
            strength=str(row.get("strength", "")),
            type=str(row.get("type", "")),
        )
        
        # Create combined judgment request (both complexity and faithfulness)
        # We'll do complexity first, then faithfulness
        # For simplicity, submit as two separate requests per triple
        
        # Complexity request
        complexity_req = model.build_judgment_request(
            passage_text, triple, passage_id, idx, "complexity"
        )
        requests.append(complexity_req)
        triple_metadata.append({
            "triple_id": triple_id,
            "passage_id": passage_id,
            "judgment_type": "complexity",
            "index": len(requests) - 1,
        })
        
        # Faithfulness request
        faithfulness_req = model.build_judgment_request(
            passage_text, triple, passage_id, idx, "faithfulness"
        )
        requests.append(faithfulness_req)
        triple_metadata.append({
            "triple_id": triple_id,
            "passage_id": passage_id,
            "judgment_type": "faithfulness",
            "index": len(requests) - 1,
        })
    
    # Submit batch job
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"judgment_{len(todo)}_triples_{timestamp}"
    
    print(f"\n🚀 Submitting batch job: {display_name}")
    print(f"   Total requests: {len(requests)} (2 per triple)")
    batch_job = model.submit_batch(requests, display_name=display_name)
    
    # Save batch state
    job_state = BatchJobState(
        job_name=batch_job.name,
        display_name=display_name,
        job_type="judgment",
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
    return poll_and_retrieve_judgment_batch(
        job_state, model, state_manager, triple_metadata, existing_records
    )


def poll_and_retrieve_judgment_batches(
    pending_jobs: list[BatchJobState],
    model,
    state_manager: BatchStateManager,
    passages_df: pd.DataFrame,
    extractions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Handle resuming multiple pending judgment batches."""
    # For simplicity, just poll the first one
    # In practice, you might want to poll all of them
    job_state = pending_jobs[0]
    
    # We need to reconstruct triple_metadata from the batch
    # For now, we'll return an error if we can't resume properly
    print("⚠️  Resuming in-progress batches not fully implemented yet.")
    print("   Please wait for batch to complete or cancel it.")
    raise NotImplementedError("Resume of in-progress judgment batches not yet supported")


def poll_and_retrieve_judgment_batch(
    job_state: BatchJobState,
    model,
    state_manager: BatchStateManager,
    triple_metadata: list[dict],
    existing_records: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Poll a judgment batch job until completion and retrieve results.
    
    Args:
        job_state: Batch job state to poll
        model: LLM model instance
        state_manager: Batch state manager
        triple_metadata: Metadata for mapping responses back to triples
        existing_records: Existing judgment records (for resume)
    
    Returns:
        DataFrame with judgment results
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
            batch_job = model.get_batch_status(job_state.job_name)
            
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
    
    # Parse results into judgment records
    print(f"\n🔍 Parsing judgment results...")
    new_records = parse_batch_judgment_results(inlined_responses, triple_metadata)
    
    # Combine with existing records
    all_records = existing_records + new_records
    result_df = pd.DataFrame(all_records)
    
    # Save to CSV
    result_df.to_csv(JUDGE_SCORES_CSV, index=False)
    logger.info("Wrote %d judge scores to %s", len(all_records), JUDGE_SCORES_CSV)
    
    print(f"\n✅ Judgment complete!")
    print(f"   Total scores: {len(all_records)}")
    print(f"   New scores: {len(new_records)}")
    print(f"   Output: {JUDGE_SCORES_CSV}")
    
    return result_df


def parse_batch_judgment_results(
    inlined_responses: list[types.InlinedResponse],
    triple_metadata: list[dict],
) -> list[dict]:
    """
    Parse batch judgment responses into judgment records.
    
    Each triple has 2 requests (complexity + faithfulness), so we need to
    pair them up before creating records.
    
    Args:
        inlined_responses: List of InlinedResponse from batch job
        triple_metadata: Metadata mapping responses to triples
    
    Returns:
        List of judgment record dictionaries
    """
    from src.models.schemas import ComplexityJudgmentSchema, FaithfulnessJudgmentSchema
    from pydantic import ValidationError
    
    # Group responses by triple_id
    triple_results = {}
    
    for idx, response_obj in enumerate(tqdm(inlined_responses, desc="Parsing responses")):
        if idx >= len(triple_metadata):
            logger.warning("Response index %d exceeds metadata length %d", idx, len(triple_metadata))
            continue
        
        meta = triple_metadata[idx]
        triple_id = meta["triple_id"]
        judgment_type = meta["judgment_type"]
        
        # Initialize triple result if needed
        if triple_id not in triple_results:
            triple_results[triple_id] = {
                "triple_id": triple_id,
                "passage_id": meta["passage_id"],
                "llm_complexity_score": -1,
                "llm_faithful": -1,
                "llm_failure_mode": "",
            }
        
        # Handle errors
        if response_obj.error:
            logger.error("Batch request failed for %s (%s): %s", 
                        triple_id, judgment_type, response_obj.error)
            continue
        
        # Parse response
        response = response_obj.response
        if not response or not response.text:
            logger.warning("Empty response for %s (%s)", triple_id, judgment_type)
            continue
        
        # Validate and parse
        try:
            if judgment_type == "complexity":
                judgment = ComplexityJudgmentSchema.model_validate_json(response.text)
                triple_results[triple_id]["llm_complexity_score"] = judgment.complexity_score
            
            elif judgment_type == "faithfulness":
                judgment = FaithfulnessJudgmentSchema.model_validate_json(response.text)
                triple_results[triple_id]["llm_faithful"] = judgment.faithful
                triple_results[triple_id]["llm_failure_mode"] = judgment.failure_mode or ""
        
        except ValidationError as exc:
            logger.error("Validation failed for %s (%s): %s", triple_id, judgment_type, exc)
        
        except Exception as exc:
            logger.error("Parse error for %s (%s): %s", triple_id, judgment_type, exc)
    
    return list(triple_results.values())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge scoring.")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini or openai")
    parser.add_argument(
        "--poll", action="store_true",
        help="Resume polling in-progress batch jobs (retrieves results if complete)."
    )
    parser.add_argument(
        "--skip-judged", action="store_true",
        help="Skip triples already in judge_scores.csv and only judge remaining triples."
    )
    args = parser.parse_args()

    for path, name in [(PASSAGES_CSV, "passages.csv"), (EXTRACTIONS_CSV, "extractions.csv")]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run the appropriate pipeline step first.")
            return

    passages_df = pd.read_csv(PASSAGES_CSV)
    extractions_df = pd.read_csv(EXTRACTIONS_CSV)

    # Run judgment via Batch API
    logger.info("Running judgment via Gemini Batch API (asynchronous processing)")
    run_judge_batch_api(passages_df, extractions_df, provider=args.provider, resume=args.resume)


if __name__ == "__main__":
    main()
