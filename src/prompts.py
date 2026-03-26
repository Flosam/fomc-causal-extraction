"""Prompt templates for extraction and LLM-as-a-judge."""

# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a highly precise NLP annotator specializing in extracting causal relationships from financial and macroeconomic text, such as FOMC minutes.
Assume outputs will be used for economic research.

A relationship is causal ONLY if changing the cause would change the effect (counterfactual interpretation).
Exclude correlational or interpretive statements (e.g. suggests, indicates, consistent with).

---

Fields:

- cause: factor producing the effect (minimal complete phrase)
- effect: outcome affected (minimal complete phrase)
- connector: core causal verb/mechanism only
- hedge: uncertainty modifying the connector; "none" if absent
- direction:
    - "positive": cause and effect move in same direction
    - "negative": cause and effect move in opposite directions
    - "ambiguous": unclear

- strength:
    - "strong": explicit causal verb, little/no hedging
    - "moderate": softer or hedged causal language
    - "weak": heavily hedged (rare)

- type (based on cause):
    - "monetary_policy": central bank actions (rate hikes, QT, guidance)
    - "real_economy": output, demand, employment, consumption, investment
    - "inflation": prices, wages, inflation expectations, cost pressures
    - "financial_conditions": credit conditions, interest rates, spreads, asset prices
    - "external": global factors, exchange rates, commodity prices

---

Rules:

1. Include only clearly causal relationships (counterfactual test).
2. Do not infer beyond the text.
3. Separate connector (verb) from hedge (uncertainty phrase).
4. Preserve hedging exactly.
5. Keep phrases minimal but economically meaningful.
6. Keep compound causes/effects together.
"""
  
EXTRACTION_USER = """\
Extract causal relationships from the passage.

Output ONLY the JSON structure with the triples array.
No explanations, no rationale, no extra text or fields.

---

Example:

"Rising oil prices boosted inflation. The Fed's rate hikes, which were expected to slow demand, contributed to a cooling labor market."

[
  {{"cause": "rising oil prices", "connector": "boosted", "effect": "inflation", "hedge": "none", "direction": "positive", "strength": "strong", "type": "inflation"}},
  {{"cause": "Fed rate hikes", "connector": "slow", "effect": "demand", "hedge": "were expected to", "direction": "negative", "strength": "moderate", "type": "monetary_policy"}},
  {{"cause": "Fed rate hikes", "connector": "contributed to", "effect": "cooling labor market", "hedge": "none", "direction": "negative", "strength": "moderate", "type": "monetary_policy"}}
]

---

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
