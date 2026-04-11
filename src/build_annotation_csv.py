"""
Annotation CSV builder.

Merges outputs/passages.csv + outputs/extractions.csv + outputs/judge_scores.csv (optional)
into outputs/annotated.csv.

If judge_scores.csv exists, LLM judge columns are pre-filled. Otherwise, they are 
left empty. Two empty columns are added for manual human override:
  - human_complexity_score  (integer 1–5, or blank)
  - human_faithful          (0 or 1, or blank)

Human values take precedence over LLM judge values in the analysis notebook.

Usage:
    python -m src.build_annotation_csv
"""

import csv
import logging
from pathlib import Path

import pandas as pd

from src.config import CONFIG, ROOT

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"
EXTRACTIONS_CSV = ROOT / "outputs" / "extractions.csv"
JUDGE_SCORES_CSV = ROOT / "outputs" / "judge_scores.csv"
ANNOTATED_CSV = ROOT / "outputs" / "annotated.csv"


def build_annotation_csv() -> pd.DataFrame:
    # Check required files (judge_scores.csv is optional)
    for path, name in [
        (PASSAGES_CSV, "passages.csv"),
        (EXTRACTIONS_CSV, "extractions.csv"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. Run the pipeline steps first."
            )

    df = pd.read_csv(EXTRACTIONS_CSV)

    # Reorder columns for readability
    col_order = [
        "triple_id", "passage_id", "meeting_date", "period", "section",
        "text",
        "cause", "connector", "effect", "hedge", "direction",
        "extraction_error",
        "human_complexity_score", "human_faithful", "human_failure_mode"
    ]

    # Merge with judge scores (optional - only if file exists)
    if JUDGE_SCORES_CSV.exists():
        judge = pd.read_csv(JUDGE_SCORES_CSV)
        df = df.merge(judge, on=["triple_id", "passage_id"], how="left")
        llm_cols = ["llm_complexity_score", "llm_faithful", "llm_failure_mode"]
        col_order = col_order + llm_cols
        logger.info("Merged judge scores from %s", JUDGE_SCORES_CSV)
    else:
        logger.warning("No judge scores found at %s", JUDGE_SCORES_CSV)

    # Add empty human override columns
    df["human_complexity_score"] = pd.NA   # fill in: integer 1–5
    df["human_faithful"] = pd.NA           # fill in: 0 or 1
    df["human_failure_mode"] = pd.NA       # fill in: failure mode label

    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # sample number of period passages for annotation (if configured)
    if CONFIG["passage"]["max_passages_per_period"] is not None:
        max_passages = CONFIG["passage"]["max_passages_per_period"]
        p1 = df[df['period'] == 'great_moderation']
        p2 = df[df['period'] == 'post_crisis_zlb']
        p3 = df[df['period'] == 'post_covid']

        passages1 = pd.Series(p1['passage_id'].unique()).sample(n=min(max_passages, p1['passage_id'].nunique()), random_state=42)
        passages2 = pd.Series(p2['passage_id'].unique()).sample(n=min(max_passages, p2['passage_id'].nunique()), random_state=42)
        passages3 = pd.Series(p3['passage_id'].unique()).sample(n=min(max_passages, p3['passage_id'].nunique()), random_state=42)

        p1_sample = p1[p1["passage_id"].isin(passages1)]
        p2_sample = p2[p2["passage_id"].isin(passages2)]
        p3_sample = p3[p3["passage_id"].isin(passages3)]
        df = pd.concat([p1_sample, p2_sample, p3_sample])

    ANNOTATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANNOTATED_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Annotation CSV written: %d rows -> %s", len(df), ANNOTATED_CSV)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_annotation_csv()


if __name__ == "__main__":
    main()
