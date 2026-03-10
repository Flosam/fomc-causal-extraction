"""Prompt templates for extraction and LLM-as-a-judge."""

# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = (
    "You are an expert NLP annotator specialising in causal relation extraction "
    "from financial and macroeconomic text."
)

EXTRACTION_USER = """\
Extract ALL causal relationships from the passage below.

Rules:
- Include only genuine causal relationships, NOT mere correlations or co-occurrences.
- Preserve hedging language exactly as written (e.g. "appeared to", "was expected to").
- Do NOT infer causality that is not stated or strongly implied in the text.
- A single passage may contain zero, one, or multiple causal relationships.

For each causal relationship return a JSON object with exactly these fields:
  "cause"     : the cause phrase (string)
  "connector" : the verbatim causal connective word/phrase (e.g. "led to", "boosted")
  "effect"    : the effect phrase (string)
  "hedge"     : any hedging language modifying the relationship, or "" if none
  "direction" : one of "positive", "negative", or "ambiguous"

Return a JSON array of these objects (empty array [] if no causal relationships found).
Return ONLY the JSON array, no commentary.

Passage:
{passage}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt — complexity scoring
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY_JUDGE_SYSTEM = (
    "You are an expert NLP annotator rating the linguistic complexity of "
    "passages from Federal Reserve meeting minutes."
)

COMPLEXITY_JUDGE_USER = """\
Rate the complexity of the following passage on a scale of 1 to 5.

Complexity criteria:
  1 – Explicit, simple causality; plain language; no hedging.
  2 – Mostly explicit causality; minimal hedging or domain jargon.
  3 – Mix of explicit and implicit causality; moderate hedging or jargon.
  4 – Predominantly implicit causality; heavy hedging; significant domain vocabulary.
  5 – Highly implicit causality; pervasive hedging; dense domain vocabulary; ambiguous direction.

Return a JSON object with exactly these fields:
  "complexity_score"     : integer 1–5
  "complexity_rationale" : one sentence explaining the score

Return ONLY the JSON object, no commentary.

Passage:
{passage}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt — faithfulness scoring
# ─────────────────────────────────────────────────────────────────────────────

FAITHFULNESS_JUDGE_SYSTEM = (
    "You are an expert NLP annotator evaluating whether a causal triple "
    "faithfully represents the source passage."
)

FAITHFULNESS_JUDGE_USER = """\
Given the source passage and the extracted causal triple below, judge whether the triple is FAITHFUL.

A triple is FAITHFUL if:
- Both the cause and effect are present in the passage (or directly entailed).
- The causal connector is accurate and not overstated.
- The direction (positive/negative/ambiguous) matches the passage.
- Any hedging in the passage is reflected in the "hedge" field (not silently dropped).

If the triple is UNFAITHFUL, identify the failure mode from this list:
  Hallucination        – cause or effect not present in the source
  Hedge Failure        – hedging present in the source but omitted or flattened
  Overclaiming         – directional claim is stronger than the source supports
  Scope Error          – cause or effect is mis-scoped (wrong subject or object)
  Correlation/Causation – a correlation is stated as causation
  Other                – any other faithfulness problem

Return a JSON object with exactly these fields:
  "faithful"             : 1 (faithful) or 0 (unfaithful)
  "faithful_rationale"   : one sentence explaining the judgment
  "failure_mode"         : failure mode label if unfaithful, or "" if faithful

Return ONLY the JSON object, no commentary.

Source passage:
{passage}

Extracted triple:
  cause     : {cause}
  connector : {connector}
  effect    : {effect}
  hedge     : {hedge}
  direction : {direction}
"""


def format_extraction(passage: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for extraction."""
    return EXTRACTION_SYSTEM, EXTRACTION_USER.format(passage=passage)


def format_complexity_judge(passage: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for complexity scoring."""
    return COMPLEXITY_JUDGE_SYSTEM, COMPLEXITY_JUDGE_USER.format(passage=passage)


def format_faithfulness_judge(passage: str, cause: str, connector: str,
                               effect: str, hedge: str, direction: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for faithfulness scoring."""
    return FAITHFULNESS_JUDGE_SYSTEM, FAITHFULNESS_JUDGE_USER.format(
        passage=passage,
        cause=cause,
        connector=connector,
        effect=effect,
        hedge=hedge,
        direction=direction,
    )
