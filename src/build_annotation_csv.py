"""
Annotation CSV builder.

Merges outputs/passages.csv + outputs/extractions.csv + outputs/judge_scores.csv
into outputs/annotated.csv.

LLM judge columns are pre-filled. Two empty columns are added for manual
human override:
  - human_complexity_score  (integer 1–5, or blank)
  - human_faithful          (0 or 1, or blank)

Human values take precedence over LLM judge values in the analysis notebook.

Usage:
    python -m src.build_annotation_csv
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import ROOT

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
JUDGE_SCORES_CSV = ROOT / "outputs" / "judge_scores.csv"
ANNOTATED_CSV = ROOT / "outputs" / "annotated.csv"


def build_annotation_csv() -> pd.DataFrame:
    for path, name in [
        (PASSAGES_CSV, "passages.csv"),
        (EXTRACTIONS_CSV, "extractions.csv"),
        (JUDGE_SCORES_CSV, "judge_scores.csv"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. Run the pipeline steps first."
            )

    passages = pd.read_csv(PASSAGES_CSV).rename(columns={"text": "passage_text"})
    extractions = pd.read_csv(EXTRACTIONS_CSV)
    judge = pd.read_csv(JUDGE_SCORES_CSV)

    # Merge extractions with passage text
    df = extractions.merge(
        passages[["passage_id", "passage_text", "period", "meeting_date", "section"]],
        on="passage_id",
        how="left",
        suffixes=("", "_passage"),
    )

    # Merge with judge scores
    df = df.merge(judge, on=["triple_id", "passage_id"], how="left")

    # Add empty human override columns
    df["human_complexity_score"] = pd.NA   # fill in: integer 1–5
    df["human_faithful"] = pd.NA           # fill in: 0 or 1

    # Reorder columns for readability
    col_order = [
        "triple_id", "passage_id", "meeting_date", "period", "section",
        "passage_text",
        "cause", "connector", "effect", "hedge", "direction",
        "llm_complexity_score",
        "llm_faithful", "llm_failure_mode",
        "human_complexity_score", "human_faithful",
        "extraction_error", "raw_response",
    ]
    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    remaining = [c for c in df.columns if c not in col_order]
    df = df[col_order + remaining]

    ANNOTATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANNOTATED_CSV, index=False)
    logger.info("Annotation CSV written: %d rows → %s", len(df), ANNOTATED_CSV)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_annotation_csv()
    print(f"\nAnnotation CSV: {len(df)} rows → {ANNOTATED_CSV}")
    print("\nFirst 3 rows (key columns):")
    preview_cols = [c for c in ["triple_id", "period", "cause", "effect",
                                 "llm_complexity_score", "llm_faithful",
                                 "human_complexity_score", "human_faithful"] if c in df.columns]
    print(df[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
