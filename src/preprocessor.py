"""
Preprocessing pipeline.

Reads sampled_meetings.csv, extracts target sections from each document
by header matching, segments each section into 3–5 sentence passages using
NLTK, and writes outputs/passages.csv.

Usage:
    python -m src.preprocessor
"""

import logging
import re
from pathlib import Path

import nltk
import pandas as pd

from src.config import CONFIG, ROOT

logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer is available
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLED_CSV = PROCESSED_DIR / "sampled_meetings.csv"
PASSAGES_CSV = ROOT / "outputs" / "passages.csv"

TARGET_SECTIONS: list[str] = CONFIG["target_sections"]
MIN_SENT = CONFIG["passage"]["min_sentences"]
MAX_SENT = CONFIG["passage"]["max_sentences"]


# ── Section extraction ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Collapse whitespace and lower-case for fuzzy header matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _strip_preamble(text: str) -> str:
    """
    Remove the administrative preamble (attendee list, procedural votes) from
    old-format FOMC minutes that have no section headers.  Finds the first
    paragraph that is substantive economic prose (long, no attendee-list
    name/title patterns) and returns everything from there onward.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    preamble_re = re.compile(
        r"\b(Mr\.|Ms\.|President|Vice\s+President|Chairman|Vice\s+Chairman"
        r"|unanimous\s+vote|were\s+approved|Secretary|General\s+Counsel)\b"
    )
    for i, para in enumerate(paragraphs):
        if len(para) > 200 and not preamble_re.search(para[:300]) and para[0].isupper():
            return "\n\n".join(paragraphs[i:])
    return text  # fallback: return everything


def _extract_sections(document_text: str) -> list[tuple[str, str]]:
    """
    Return list of (section_label, section_text) for each target section found.

    Strategy: split on lines that look like section headers (all-caps or
    title-cased, possibly preceded by a roman numeral or letter), then keep
    sections whose normalised header matches a target.
    """
    # Split into lines; identify header candidates
    lines = document_text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_header: str | None = None
    current_body: list[str] = []

    header_re = re.compile(
        r"^(?:[IVXivx]+\.\s+|[A-Z]\.\s+)?[A-Z][A-Za-z\s,\'\-]{10,}$"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_header is not None:
                current_body.append("")
            continue
        # Heuristic: a header is a short line (≤ 120 chars) that matches the pattern
        if len(stripped) <= 120 and header_re.match(stripped):
            if current_header is not None:
                sections.append((current_header, current_body))
            current_header = stripped
            current_body = []
        else:
            if current_header is not None:
                current_body.append(stripped)

    if current_header is not None:
        sections.append((current_header, current_body))

    # Filter to target sections
    target_normalised = [_normalize(t) for t in TARGET_SECTIONS]
    matched: list[tuple[str, str]] = []
    for header, body_lines in sections:
        nh = _normalize(header)
        # Substring match: header contains a target substring or vice-versa
        if any(t in nh or nh in t for t in target_normalised):
            matched.append((header, "\n".join(body_lines)))

    return matched


# ── Passage segmentation ──────────────────────────────────────────────────────

def _segment_into_passages(section_text: str) -> list[str]:
    """
    Tokenise section_text into sentences, then group into non-overlapping
    windows of up to MAX_SENT sentences. All windows are kept — even a
    single-sentence window can contain a causal triple.
    """
    sentences = nltk.sent_tokenize(section_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    passages = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i: i + MAX_SENT]
        if len(chunk) >= MIN_SENT:   # MIN_SENT == 1, so this always passes
            passages.append(" ".join(chunk))
        i += MAX_SENT  # non-overlapping windows

    return passages


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_passages(sampled_csv: Path = SAMPLED_CSV) -> pd.DataFrame:
    _ensure_nltk()

    df = pd.read_csv(sampled_csv, parse_dates=["date"])
    records = []
    passage_id = 0

    for _, row in df.iterrows():
        doc_text: str = str(row.get("text", ""))
        if not doc_text.strip():
            logger.warning("Empty document for meeting %s — skipping.", row.get("date"))
            continue

        sections = _extract_sections(doc_text)
        if not sections:
            logger.warning(
                "No target sections found in meeting %s — falling back to full text.",
                row.get("date"),
            )
            sections = [("full_document", _strip_preamble(doc_text))]

        for section_label, section_text in sections:
            passages = _segment_into_passages(section_text)
            for passage_text in passages:
                records.append({
                    "passage_id": f"p{passage_id:05d}",
                    "meeting_date": row["date"].date(),
                    "year": row["date"].year,
                    "period": row.get("period", ""),
                    "section": section_label,
                    "text": passage_text,
                })
                passage_id += 1

    result = pd.DataFrame(records)
    output_path = PASSAGES_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(
        "Wrote %d passages from %d meetings to %s",
        len(result), len(df), output_path,
    )
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_passages()
    print(df.groupby("period")["passage_id"].count().rename("passages").to_string())
    print(f"\nTotal: {len(df)} passages → {PASSAGES_CSV}")


if __name__ == "__main__":
    main()
