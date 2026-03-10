"""
LLM-as-a-judge orchestration.

Reads outputs/passages.csv + outputs/extractions.csv, calls the judge model
to score each triple for complexity (1–5) and faithfulness (0/1), and writes
outputs/judge_scores.csv.

Usage:
    python -m src.judge
    python -m src.judge --provider openai
    python -m src.judge --resume
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import ROOT
from src.models import get_model

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
JUDGE_SCORES_CSV = ROOT / "outputs" / "judge_scores.csv"


def run_judge(
    passages_df: pd.DataFrame,
    extractions_df: pd.DataFrame,
    provider: str | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """
    For each non-error triple in *extractions_df*, call the judge model to score
    complexity and faithfulness. Returns a DataFrame with one row per triple.
    """
    from src.models.base import CausalTriple

    model = get_model(provider)

    # Index passages by passage_id for quick lookup
    passage_lookup = passages_df.set_index("passage_id")["text"].to_dict()

    # Filter out null-extraction rows
    valid = extractions_df[extractions_df["extraction_error"] != "no_triples"].copy()
    valid = valid[valid["cause"].astype(str).str.strip() != ""]

    # Load existing results if resuming
    existing_ids: set[str] = set()
    existing_records: list[dict] = []
    if resume and JUDGE_SCORES_CSV.exists():
        existing_df = pd.read_csv(JUDGE_SCORES_CSV)
        existing_ids = set(existing_df["triple_id"].astype(str))
        existing_records = existing_df.to_dict("records")
        logger.info("Resuming — skipping %d already-judged triple IDs", len(existing_ids))

    todo = valid[~valid["triple_id"].astype(str).isin(existing_ids)]
    logger.info("Judging %d triples…", len(todo))

    new_records: list[dict] = []

    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Judging"):
        passage_id = str(row["passage_id"])
        triple_id = str(row["triple_id"])
        passage_text = passage_lookup.get(passage_id, "")

        triple = CausalTriple(
            cause=str(row["cause"]),
            connector=str(row["connector"]),
            effect=str(row["effect"]),
            hedge=str(row["hedge"]),
            direction=str(row["direction"]),
        )

        try:
            result = model.judge(passage_text, triple)
            new_records.append({
                "triple_id": triple_id,
                "passage_id": passage_id,
                "llm_complexity_score": result.complexity_score,
                "llm_complexity_rationale": result.complexity_rationale,
                "llm_faithful": result.faithful,
                "llm_faithful_rationale": result.faithful_rationale,
                "llm_failure_mode": result.failure_mode,
            })
        except Exception as exc:
            logger.error("Judge failed for triple %s: %s", triple_id, exc)
            new_records.append({
                "triple_id": triple_id,
                "passage_id": passage_id,
                "llm_complexity_score": -1,
                "llm_complexity_rationale": f"error: {exc}",
                "llm_faithful": -1,
                "llm_faithful_rationale": f"error: {exc}",
                "llm_failure_mode": "Error",
            })

    all_records = existing_records + new_records
    result_df = pd.DataFrame(all_records)
    JUDGE_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(JUDGE_SCORES_CSV, index=False)
    logger.info("Wrote %d judge scores to %s", len(result_df), JUDGE_SCORES_CSV)
    return result_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge scoring.")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini or openai")
    parser.add_argument("--resume", action="store_true",
                        help="Skip triples already present in judge_scores.csv.")
    args = parser.parse_args()

    for path, name in [(PASSAGES_CSV, "passages.csv"), (EXTRACTIONS_CSV, "extractions.csv")]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run the appropriate pipeline step first.")
            return

    passages_df = pd.read_csv(PASSAGES_CSV)
    extractions_df = pd.read_csv(EXTRACTIONS_CSV)

    run_judge(passages_df, extractions_df, provider=args.provider, resume=args.resume)


if __name__ == "__main__":
    main()
