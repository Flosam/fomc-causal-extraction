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
    """
    One extracted causal relationship with full extraction metadata.
    
    This class represents a single extraction result, including the core triple
    fields, source context, error information, and token usage. Used for both
    successful extractions and error records.
    
    Fields:
        cause: The factor that produces the effect
        connector: Causal mechanism/verb (e.g., 'increased', 'led to')
        effect: The outcome being affected
        hedge: Hedging/uncertainty language (empty string if none)
        direction: Causal direction ("positive" | "negative" | "ambiguous")
        triple_id: Unique identifier for this record
        passage_id: ID of the source passage
        model: Name of the model used for extraction
        meeting_date: FOMC meeting date (populated during merge)
        period: Economic period label (populated during merge)
        text: Source passage text (populated during merge)
        raw_response: Full LLM response for debugging
        extraction_error: Error message (empty string if successful)
        malformed_count: Number of malformed items in response
        prompt_tokens: Input tokens used for this extraction
        completion_tokens: Output tokens generated for this extraction
    """
    cause: str
    connector: str
    effect: str
    hedge: str
    direction: str
    
    # Identifiers
    triple_id: str = ""
    passage_id: str = ""
    model: str = ""
    
    # Source context (populated during merge with passages.csv)
    meeting_date: Optional[str] = None
    period: Optional[str] = None
    text: Optional[str] = None
    
    # Extraction metadata
    raw_response: str = ""
    extraction_error: str = ""
    malformed_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV output."""
        from dataclasses import asdict
        return asdict(self)


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
