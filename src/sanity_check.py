"""
Sanity check: verify that passages derived from the corpus contain
the 12 canonical economic causal relationships specified in config.yaml.

Usage:
    python -m src.sanity_check
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import CONFIG, ROOT

logger = logging.getLogger(__name__)

PASSAGES_CSV = ROOT / "outputs" / "passages.csv"


def _passage_contains_pair(text: str, cause_kw: str, effect_kw: str) -> bool:
    """Loose keyword check — both keywords must appear (case-insensitive)."""
    text_lower = text.lower()
    return cause_kw.lower() in text_lower and effect_kw.lower() in text_lower


def run_sanity_check(passages_csv: Path = PASSAGES_CSV) -> dict:
    """
    For each canonical pair, check if at least one passage contains both
    the cause keyword and the effect keyword.

    Returns a dict: {pair_label -> bool}
    """
    df = pd.read_csv(passages_csv)
    all_text = " ".join(df["text"].astype(str).tolist()).lower()

    pairs = CONFIG.get("canonical_causal_pairs", [])
    results: dict[str, bool] = {}

    for pair in pairs:
        cause = pair["cause"]
        effect = pair["effect"]
        label = f"{cause} → {effect}"
        found = any(
            _passage_contains_pair(row["text"], cause, effect)
            for _, row in df.iterrows()
        )
        results[label] = found

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not PASSAGES_CSV.exists():
        print(f"ERROR: {PASSAGES_CSV} not found. Run src/preprocessor.py first.")
        return

    results = run_sanity_check()

    passed = sum(v for v in results.values())
    total = len(results)

    print(f"\nSanity Check Results: {passed}/{total} canonical pairs found in corpus\n")
    print(f"{'Pair':<55} {'Found'}")
    print("-" * 65)
    for label, found in results.items():
        status = "✓" if found else "✗  <-- MISSING"
        print(f"  {label:<55} {status}")

    if passed < total:
        print(
            f"\nWARNING: {total - passed} canonical pair(s) not found. "
            "Consider reviewing section extraction or sampling."
        )
    else:
        print("\nAll canonical pairs present — corpus sanity check passed.")


if __name__ == "__main__":
    main()
