"""
Extraction orchestration.

Reads outputs/passages.csv, calls the configured LLM to extract causal triples
from each passage, and writes outputs/extractions.csv.

Usage:
    python -m src.extractor
    python -m src.extractor --provider openai   # override config.yaml
    python -m src.extractor --held-out-only     # run only the 20-passage refinement set
    python -m src.extractor --resume            # skip passages already in extractions.csv
    python -m src.extractor --batch-size 15     # concurrent requests (default: 10)
"""

import argparse
import asyncio
import logging
import signal
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import CONFIG, ROOT
from src.models import get_model
from src.prompts import EXTRACTION_SYSTEM, format_extraction

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
HELD_OUT_CSV = ROOT / "outputs" / "held_out_passages.csv"

HELD_OUT_SIZE: int = CONFIG.get("held_out_size", 20)

# ── Interrupt handling ────────────────────────────────────────────────────────

# Global flag to track if user interrupted with Ctrl+C
_interrupt_received = False

def _handle_interrupt(signum, frame):
    """Signal handler for Ctrl+C (SIGINT)."""
    global _interrupt_received
    if not _interrupt_received:  # Only log once
        _interrupt_received = True
        logger.warning("\n⚠️  Interrupt received (Ctrl+C). Saving progress and exiting gracefully...")
        logger.info("Current work will be saved. You can resume later with --resume flag.")


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
    sample.to_csv(HELD_OUT_CSV, index=False)
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
    resume: bool = False,
) -> pd.DataFrame:
    """
    Extract causal triples for all rows in *passages_df* (SEQUENTIAL).

    DEPRECATED: Use run_extraction_batch() for 5-10x faster processing.
    Returns a DataFrame with one row per extracted triple.
    """
    model = get_model(provider)

    # Load existing results if resuming
    existing_ids: set[str] = set()
    existing_records: list[dict] = []
    if resume and EXTRACTIONS_CSV.exists():
        existing_df = pd.read_csv(EXTRACTIONS_CSV)
        existing_ids = set(existing_df["passage_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info("Resuming — skipping %d already-processed passage IDs", len(existing_ids))

    todo = passages_df[~passages_df["passage_id"].astype(str).isin(existing_ids)]
    
    new_records: list[dict] = []
    triple_counter = 0
    save_interval = 5  # Save every 5 passages (reduced from 10 for better crash resilience)
    
    logger.info("Processing %d passages (SEQUENTIAL MODE)…", len(todo))
    logger.info("Auto-saving every %d passages. Press Ctrl+C to safely interrupt.", save_interval)

    # Register interrupt handler
    global _interrupt_received
    _interrupt_received = False
    signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        for idx, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="Extracting"), 1):
            # Check for interrupt before processing
            if _interrupt_received:
                logger.info("Stopping due to interrupt. Processed %d/%d passages.", idx - 1, len(todo))
                break
            
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
                    "strength": "",
                    "type": "",
                    "raw_response": "",
                    "extraction_error": "no_triples",
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
                        "strength": triple.strength,
                        "type": triple.type,
                        "raw_response": triple.raw_response,
                        "extraction_error": "",
                    })
                    triple_counter += 1
            
            # Save progress every save_interval passages
            if idx % save_interval == 0:
                all_records = existing_records + new_records
                result = pd.DataFrame(all_records)
                EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(EXTRACTIONS_CSV, index=False)
                logger.info("✓ Auto-saved progress: %d passages processed", idx)
    
    finally:
        # Always save final state, even on interrupt or error
        if new_records:
            all_records = existing_records + new_records
            result = pd.DataFrame(all_records)
            EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(EXTRACTIONS_CSV, index=False)
            
            if _interrupt_received:
                logger.info("✓ Saved %d passages before exiting. Resume with --resume flag.", len(new_records))
            else:
                logger.info(
                    "Wrote %d triples (%d passages) to %s",
                    triple_counter, len(todo), EXTRACTIONS_CSV,
                )
    
    return result if 'result' in locals() else pd.DataFrame(existing_records)


# ── Extraction loop (batch/concurrent version) ───────────────────────────────

async def run_extraction_batch(
    passages_df: pd.DataFrame,
    provider: str | None = None,
    resume: bool = False,
    batch_size: int = 10,
    batch_delay: float = 0.0,
) -> pd.DataFrame:
    """
    Extract causal triples with CONCURRENT API calls for 5-10x speedup.

    Args:
        passages_df: DataFrame with passages to extract from
        provider: LLM provider (gemini, openai, etc.)
        resume: Skip passages already in extractions.csv
        batch_size: Number of concurrent API requests (default: 10)
        batch_delay: Seconds to wait between batches to respect rate limits (default: 0)

    Returns:
        DataFrame with one row per extracted triple
    """
    model = get_model(provider)

    # Check if model supports async extraction
    logger.debug(f"Model type: {type(model)}, has extract_async: {hasattr(model, 'extract_async')}")
    if not hasattr(model, 'extract_async'):
        logger.warning("Model does not support async extraction, falling back to sequential mode")
        return run_extraction(passages_df, provider, resume)

    # Load existing results if resuming
    existing_ids: set[str] = set()
    existing_records: list[dict] = []
    if resume and EXTRACTIONS_CSV.exists():
        existing_df = pd.read_csv(EXTRACTIONS_CSV)
        existing_ids = set(existing_df["passage_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info("Resuming — skipping %d already-processed passage IDs", len(existing_ids))

    todo = passages_df[~passages_df["passage_id"].astype(str).isin(existing_ids)]
    logger.info(
        "Processing %d passages with batch_size=%d (CONCURRENT MODE)…",
        len(todo), batch_size
    )
    logger.info("Auto-saving after each batch. Press Ctrl+C to safely interrupt.")

    # Register interrupt handler
    global _interrupt_received
    _interrupt_received = False
    signal.signal(signal.SIGINT, _handle_interrupt)

    new_records: list[dict] = []
    triple_counter = 0
    
    # Process in batches to control concurrency
    total_batches = (len(todo) + batch_size - 1) // batch_size
    
    try:
        with tqdm(total=len(todo), desc="Extracting (concurrent)") as pbar:
            for batch_idx in range(total_batches):
                # Check for interrupt before processing batch
                if _interrupt_received:
                    logger.info("Stopping due to interrupt. Processed %d/%d batches.", batch_idx, total_batches)
                    break
                
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(todo))
                batch_df = todo.iloc[batch_start:batch_end]
                
                # Create concurrent extraction tasks for this batch
                tasks = []
                batch_rows = []
            for _, row in batch_df.iterrows():
                passage_text = str(row["text"])
                tasks.append(model.extract_async(passage_text))
                batch_rows.append(row)
            
            # Execute batch concurrently
            batch_start_time = time.time()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start_time
            
            logger.debug(
                "Batch %d/%d completed in %.2fs (%.2fs/passage avg)",
                batch_idx + 1, total_batches, batch_time, batch_time / len(tasks)
            )
            
            # Add delay between batches to respect rate limits (if configured)
            if batch_delay > 0 and batch_idx < total_batches - 1:  # Don't delay after last batch
                logger.debug(f"Waiting {batch_delay}s before next batch...")
                await asyncio.sleep(batch_delay)
            
            # Process results and build records for this batch
            batch_records = []
            for row, result in zip(batch_rows, batch_results):
                passage_id = str(row["passage_id"])
                passage_text = str(row["text"])
                
                # Handle extraction errors
                if isinstance(result, Exception):
                    logger.error("Extraction failed for passage %s: %s", passage_id, result)
                    batch_records.append({
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
                        "strength": "",
                        "type": "",
                        "raw_response": "",
                        "extraction_error": f"exception: {type(result).__name__}",
                    })
                    pbar.update(1)
                    continue
                
                triples = result
                
                if not triples:
                    # Record null extraction
                    batch_records.append({
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
                        "strength": "",
                        "type": "",
                        "raw_response": "",
                        "extraction_error": "no_triples",
                    })
                else:
                    # Record each triple
                    for i, triple in enumerate(triples):
                        batch_records.append({
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
                            "strength": triple.strength,
                            "type": triple.type,
                            "raw_response": triple.raw_response,
                            "extraction_error": "",
                        })
                        triple_counter += 1
                
                pbar.update(1)
            
            # Save progress after each batch (incremental save)
            new_records.extend(batch_records)
            all_records = existing_records + new_records
            result = pd.DataFrame(all_records)
            EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(EXTRACTIONS_CSV, index=False)
            logger.info(
                "✓ Auto-saved batch %d/%d (%d total triples so far)",
                batch_idx + 1, total_batches, len(all_records)
            )
    
    finally:
        # Always save final state, even on interrupt or error
        if new_records:
            all_records = existing_records + new_records
            result = pd.DataFrame(all_records)
            EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(EXTRACTIONS_CSV, index=False)
            
            if _interrupt_received:
                logger.info("✓ Saved %d passages before exiting. Resume with --resume flag.", len(new_records))
            else:
                logger.info(
                    "Wrote %d triples (%d passages) to %s",
                    triple_counter, len(todo), EXTRACTIONS_CSV,
                )

    return result if 'result' in locals() else pd.DataFrame(existing_records)


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
        one row with empty cause/effect/etc and extraction_error="no_triples".
    
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
                strength=item.get("strength", ""),
                type=item.get("type", ""),
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
            "extraction_error": extraction_error or "no_triples",
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
                "strength": triple.strength,
                "type": triple.type,
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
        "--resume", action="store_true",
        help="Skip passages already present in extractions.csv.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of concurrent API requests (default: 10). Use 1 for sequential.",
    )
    parser.add_argument(
        "--batch-delay", type=float, default=0.0,
        help="Seconds to wait between batches to avoid rate limits (default: 0). "
             "Recommended: 10-15s for GitHub Models with batch-size > 1.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Force sequential processing (same as --batch-size 1).",
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

    # Choose extraction mode
    if args.sequential or args.batch_size == 1:
        logger.info("Running in SEQUENTIAL mode (no concurrency)")
        run_extraction(passages_df, provider=args.provider, resume=args.resume)
    else:
        logger.info("Running in CONCURRENT mode with batch_size=%d, batch_delay=%.1fs", 
                   args.batch_size, args.batch_delay)
        asyncio.run(run_extraction_batch(
            passages_df,
            provider=args.provider,
            resume=args.resume,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay,
        ))


if __name__ == "__main__":
    main()
