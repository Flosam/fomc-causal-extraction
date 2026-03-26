"""Batch job state persistence and management."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import ROOT

logger = logging.getLogger(__name__)

BATCH_STATE_FILE = ROOT / "outputs" / "batch_jobs.json"


@dataclass
class BatchJobState:
    """State tracking for a batch job."""
    job_name: str  # Batch job ID from API
    display_name: str  # Human-readable name
    job_type: str  # "extraction" or "judgment"
    model: str  # Model name used
    created_at: str  # ISO timestamp
    state: str  # Job state (e.g., "JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED")
    total_requests: int  # Number of requests in batch
    completed_requests: Optional[int] = None  # Number completed (from completion_stats)
    failed_requests: Optional[int] = None  # Number failed
    updated_at: Optional[str] = None  # Last poll timestamp
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "BatchJobState":
        """Create from dictionary."""
        return cls(**data)


class BatchStateManager:
    """Manages batch job state persistence."""
    
    def __init__(self, state_file: Path = BATCH_STATE_FILE):
        self.state_file = state_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create state file if it doesn't exist."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self.state_file.write_text("{}")
    
    def load_all(self) -> dict[str, BatchJobState]:
        """Load all batch job states."""
        try:
            data = json.loads(self.state_file.read_text())
            return {k: BatchJobState.from_dict(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning("Failed to load batch state file: %s", exc)
            return {}
    
    def save(self, job_state: BatchJobState):
        """Save or update a batch job state."""
        all_states = self.load_all()
        all_states[job_state.job_name] = job_state
        
        # Convert to dict for JSON
        data = {k: v.to_dict() for k, v in all_states.items()}
        
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug("Saved batch state: %s", job_state.job_name)
    
    def get(self, job_name: str) -> Optional[BatchJobState]:
        """Get a specific batch job state."""
        all_states = self.load_all()
        return all_states.get(job_name)
    
    def get_pending(self, job_type: Optional[str] = None) -> list[BatchJobState]:
        """Get all pending/running batch jobs, optionally filtered by type."""
        all_states = self.load_all()
        pending_states = ["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED",
                         "JobState.JOB_STATE_PENDING", "JobState.JOB_STATE_RUNNING", "JobState.JOB_STATE_QUEUED"]
        
        results = []
        for state in all_states.values():
            if state.state in pending_states:
                if job_type is None or state.job_type == job_type:
                    results.append(state)
        
        return results
    
    def delete(self, job_name: str):
        """Remove a batch job from state tracking."""
        all_states = self.load_all()
        if job_name in all_states:
            del all_states[job_name]
            data = {k: v.to_dict() for k, v in all_states.items()}
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("Deleted batch state: %s", job_name)
