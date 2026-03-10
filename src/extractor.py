"""
Extraction orchestration.

Reads outputs/passages.csv, calls the configured LLM to extract causal triples
from each passage, and writes outputs/extractions.csv.

Usage:
    python -m src.extractor
    python -m src.extractor --provider openai   # override config.yaml
    python -m src.extractor --held-out-only     # run only the 20-passage refinement set
    python -m src.extractor --resume            # skip passages already in extractions.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import CONFIG, ROOT
from src.models import get_model

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
HELD_OUT_CSV = ROOT / "outputs" / "held_out_passages.csv"

HELD_OUT_SIZE: int = CONFIG.get("held_out_size", 20)


# ── Held-out set management ───────────────────────────────────────────────────

def get_held_out(passages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the held-out refinement set.

    If outputs/held_out_passages.csv already exists, load it (so the same 20
    passages are used across runs). Otherwise sample HELD_OUT_SIZE rows
    uniformly and save the file.
    """
    if HELD_OUT_CSV.exists():
        return pd.read_csv(HELD_OUT_CSV)

    sample = passages_df.sample(n=min(HELD_OUT_SIZE, len(passages_df)), random_state=42)
    sample.to_csv(HELD_OUT_CSV, index=False)
    logger.info("Held-out set (%d passages) saved to %s", len(sample), HELD_OUT_CSV)
    return sample


# ── Extraction loop ───────────────────────────────────────────────────────────

def run_extraction(
    passages_df: pd.DataFrame,
    provider: str | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Extract causal triples for all rows in *passages_df*.

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
    logger.info("Processing %d passages…", len(todo))

    new_records: list[dict] = []
    triple_counter = 0

    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Extracting"):
        passage_id = str(row["passage_id"])
        passage_text = str(row["text"])

        triples = model.extract(passage_text)
        if not triples:
            # Record a null extraction so we know the passage was processed
            new_records.append({
                "triple_id": f"{passage_id}_t000",
                "passage_id": passage_id,
                "meeting_date": row.get("meeting_date", ""),
                "period": row.get("period", ""),
                "cause": "",
                "connector": "",
                "effect": "",
                "hedge": "",
                "direction": "",
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
                    "cause": triple.cause,
                    "connector": triple.connector,
                    "effect": triple.effect,
                    "hedge": triple.hedge,
                    "direction": triple.direction,
                    "raw_response": triple.raw_response,
                    "extraction_error": "",
                })
                triple_counter += 1

    all_records = existing_records + new_records
    result = pd.DataFrame(all_records)
    EXTRACTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(EXTRACTIONS_CSV, index=False)
    logger.info(
        "Wrote %d triples (%d passages) to %s",
        triple_counter, len(todo), EXTRACTIONS_CSV,
    )
    return result


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
    args = parser.parse_args()

    if not PASSAGES_CSV.exists():
        print(f"ERROR: {PASSAGES_CSV} not found. Run src/preprocessor.py first.")
        return

    passages_df = pd.read_csv(PASSAGES_CSV)

    if args.held_out_only:
        passages_df = get_held_out(passages_df)
        logger.info("Running on held-out set (%d passages).", len(passages_df))

    run_extraction(passages_df, provider=args.provider, resume=args.resume)


if __name__ == "__main__":
    main()
