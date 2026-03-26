"""Abstract base class for all LLM model adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage information from an API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CausalTriple:
    """One extracted causal relationship."""
    cause: str
    connector: str          # verbatim causal language from the source
    effect: str
    hedge: str              # hedging language, or empty string
    direction: str          # "positive" | "negative" | "ambiguous"
    strength: str = ""      # "strong" | "moderate" | "weak"
    type: str = ""          # "monetary_policy" | "real_economy" | "inflation" | "financial_conditions" | "external" | "other"
    raw_response: str = ""  # full JSON string returned by the model
    prompt_tokens: int = 0  # input tokens used for this extraction
    completion_tokens: int = 0  # output tokens generated for this extraction


@dataclass
class JudgmentResult:
    """LLM-as-judge output for a single triple."""
    complexity_score: int           # 1–5
    faithful: int                   # 1 = faithful, 0 = unfaithful
    failure_mode: str = ""          # non-empty when faithful == 0


class LLMModel(ABC):
    """
    Abstract interface for causal extraction and judgment.

    Concrete adapters (Gemini, OpenAI …) must implement:
      - extract()  — causal triple extraction from a passage
      - judge()    — complexity + faithfulness scoring of a passage/triple pair
    """

    @abstractmethod
    def extract(self, passage: str) -> list[CausalTriple]:
        """
        Extract causal triples from *passage*.

        Returns a (possibly empty) list of CausalTriple objects.
        May raise `ExtractionError` on unrecoverable failure.
        """

    @abstractmethod
    def judge(self, passage: str, triple: CausalTriple) -> JudgmentResult:
        """
        Judge the complexity of *passage* and the faithfulness of *triple*.

        Returns a JudgmentResult.
        May raise `ExtractionError` on unrecoverable failure.
        """


class ExtractionError(Exception):
    """Raised when an LLM call fails after all retries."""
