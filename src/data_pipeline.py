"""
Data acquisition pipeline.

Downloads the FOMC minutes corpus from Kaggle (vladtasca/fomc-meeting-statements-minutes),
filters to 30 meetings spread across the three economic periods defined in config.yaml,
and saves them to data/raw/.

Usage:
    python -m src.data_pipeline          # downloads + filters
    python -m src.data_pipeline --skip-download  # if CSV already in data/raw/
"""

import argparse
import logging
import os
import random
from pathlib import Path

import pandas as pd

from src.config import CONFIG, ROOT

logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Expected filename after Kaggle download
KAGGLE_CSV = RAW_DIR / "fomc_minutes.csv"
SAMPLED_CSV = PROCESSED_DIR / "sampled_meetings.csv"


# ── Download ──────────────────────────────────────────────────────────────────

def download_fomc_dataset():
    """Download the FOMC dataset from Kaggle using the kaggle CLI / API."""
    import kaggle  # imported here so missing package gives a clear error

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FOMC dataset from Kaggle…")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "vladtasca/fomc-meeting-statements-minutes",
        path=str(RAW_DIR),
        unzip=True,
    )
    # Rename to a stable filename if needed
    candidates = list(RAW_DIR.glob("*.csv"))
    if not KAGGLE_CSV.exists() and candidates:
        candidates[0].rename(KAGGLE_CSV)
    logger.info("Download complete: %s", KAGGLE_CSV)


# ── Filtering ─────────────────────────────────────────────────────────────────

def _period_for_year(year: int) -> str | None:
    for period_key, period_cfg in CONFIG["periods"].items():
        if period_cfg["start_year"] <= year <= period_cfg["end_year"]:
            return period_key
    return None


def filter_and_sample(csv_path: Path = KAGGLE_CSV, seed: int = 42) -> pd.DataFrame:
    """
    Load the raw CSV, assign each row to an economic period, and sample
    `meetings_to_sample` rows per period (stratified random sample).

    Expected columns in the Kaggle CSV:
        date, type, text   (plus possibly others — we keep them all)
    """
    logger.info("Loading corpus from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse date — try common formats
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year

    # Filter to minutes only (not statements)
    if "type" in df.columns:
        df = df[df["type"].str.lower().str.contains("minute", na=False)]

    # Assign period
    df["period"] = df["year"].apply(_period_for_year)
    df = df.dropna(subset=["period"])

    # Sample per period
    rng = random.Random(seed)
    sampled_frames = []
    for period_key, period_cfg in CONFIG["periods"].items():
        subset = df[df["period"] == period_key].copy()
        n = period_cfg["meetings_to_sample"]
        if len(subset) < n:
            logger.warning(
                "Period %r has only %d meetings (wanted %d) — using all.",
                period_key, len(subset), n,
            )
            n = len(subset)
        idx = rng.sample(list(subset.index), n)
        sampled_frames.append(subset.loc[idx])
        logger.info("Period %-20s → sampled %d meetings", period_key, n)

    result = pd.concat(sampled_frames).sort_values("date").reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(SAMPLED_CSV, index=False)
    logger.info("Saved %d sampled meetings to %s", len(result), SAMPLED_CSV)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Acquire and filter FOMC minutes corpus.")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Kaggle download (use if CSV is already in data/raw/).",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_fomc_dataset()
    else:
        if not KAGGLE_CSV.exists():
            raise FileNotFoundError(
                f"--skip-download was set but {KAGGLE_CSV} does not exist. "
                "Run without --skip-download first."
            )

    df = filter_and_sample()
    print(df[["date", "year", "period"]].to_string(index=False))
    print(f"\n{len(df)} meetings saved to {SAMPLED_CSV}")


if __name__ == "__main__":
    main()
