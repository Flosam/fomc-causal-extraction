"""Pydantic schemas for structured LLM outputs.

These schemas are used with Gemini's response_json_schema parameter to ensure
type-safe, validated responses from the LLM.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CausalTripleSchema(BaseModel):
    """Schema for a single extracted causal relationship."""
    
    cause: str = Field(
        ...,
        description="The factor that produces the effect. Minimal complete phrase. Must not be 'none' or empty.",
        min_length=1
    )
    
    connector: str = Field(
        ...,
        description="The core causal mechanism (e.g., 'increased', 'reduced', 'led to'). Causal verb only, no hedging.",
        min_length=1
    )
    
    effect: str = Field(
        ...,
        description="The outcome being affected. Minimal complete phrase. Must not be 'none' or empty.",
        min_length=1
    )
    
    hedge: str = Field(
        default="",
        description="Any uncertainty or expectation modifying the connector (e.g., 'appeared to', 'was expected to'). Empty string if no hedging present."
    )
    
    direction: Literal["positive", "negative", "ambiguous"] = Field(
        ...,
        description="Relationship direction: 'positive' if cause and effect move together, 'negative' if opposite, 'ambiguous' if unclear"
    )
    
    strength: Literal["strong", "moderate", "weak"] = Field(
        default="moderate",
        description="Causal strength: 'strong' for explicit direct causality, 'moderate' for clear but hedged, 'weak' for heavily hedged"
    )
    
    type: Literal[
        "monetary_policy",
        "real_economy", 
        "inflation",
        "financial_conditions",
        "external",
        "other"
    ] = Field(
        default="other",
        description="Economic category based on the nature of the cause: monetary_policy (central bank actions), real_economy (output, demand, employment), inflation (prices, wages), financial_conditions (credit, rates, spreads), external (global factors, commodities), other"
    )


class ExtractionResponseSchema(BaseModel):
    """Schema for the complete extraction response containing multiple triples."""
    
    triples: List[CausalTripleSchema] = Field(
        default_factory=list,
        description="List of all causal relationships found in the passage. Empty list if no causal relationships found."
    )


class ComplexityJudgmentSchema(BaseModel):
    """Schema for complexity scoring of a passage."""
    
    complexity_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Linguistic complexity score from 1 to 5, where 1 = simple explicit causality, 5 = dense hedged implicit causality"
    )
    
    complexity_rationale: str = Field(
        ...,
        description="One sentence explaining why this complexity score was assigned",
        min_length=10
    )


class FaithfulnessJudgmentSchema(BaseModel):
    """Schema for faithfulness judgment of an extracted triple."""
    
    faithful: Literal[0, 1] = Field(
        ...,
        description="1 if the extracted triple faithfully represents the source passage, 0 if unfaithful"
    )
    
    faithful_rationale: str = Field(
        ...,
        description="One sentence explaining the faithfulness judgment",
        min_length=10
    )
    
    failure_mode: Literal[
        "",
        "Hallucination",
        "Hedge Failure",
        "Overclaiming",
        "Scope Error",
        "Correlation/Causation",
        "Other"
    ] = Field(
        default="",
        description="Failure mode label if unfaithful (empty string if faithful). Options: Hallucination (cause/effect not in source), Hedge Failure (hedging omitted), Overclaiming (claim too strong), Scope Error (wrong subject/object), Correlation/Causation (correlation stated as causation), Other"
    )
