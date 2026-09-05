"""Data models for the Trace Receiver system.

Defines configuration, storage, and state models for receiving
trace data from remote AI agents and triggering evaluations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class MatchingStrategy(str, Enum):
    """How to match incoming trace.input with dataset queries."""
    EXACT = "exact"
    NORMALIZED = "normalized"


class TraceProjectConfig(BaseModel):
    """Configuration for a trace receiver project."""
    project: str
    api_key_hash: str = ""
    dataset_id: str = ""
    input_column: str = "input"
    expected_output_column: Optional[str] = None
    context_column: Optional[str] = None
    expected_tools_column: Optional[str] = None
    matching_strategy: MatchingStrategy = MatchingStrategy.NORMALIZED
    metrics: List[Dict[str, Any]] = []
    eval_model: str = "gpt-4o-mini"
    auto_evaluate: bool = True
    runs_per_query: int = 1
    trace_timeout_seconds: int = 300
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class StoredTrace(BaseModel):
    """A single trace received from a remote agent.

    Mirrors the sender payload one-to-one so nothing is dropped on ingest:
    ``usage`` is the accumulated counter block, ``metadata`` is the caller's
    declared metadata object, verbatim. ``is_partial`` marks a record that
    was assembled from streamed ``partial_span`` messages and is still
    waiting for the authoritative final trace.
    """
    trace_id: str
    project: str
    input: str = ""
    output: str = ""
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    response_time: Optional[float] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    tools_called: Optional[List[str]] = None
    spans: Optional[List[Dict[str, Any]]] = None
    span_count: int = 0
    cost_usd: Optional[float] = None
    cost_source: Optional[str] = None  # "reported" | "estimated"
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    num_turns: Optional[int] = None
    is_partial: bool = False
    received_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    matched_query_index: Optional[int] = None
    run_index: int = 0
    evaluation_status: str = "pending"
    evaluation_session_id: Optional[str] = None


class TraceProjectState(BaseModel):
    """Runtime state for a trace receiver project."""
    config: TraceProjectConfig
    traces: List[StoredTrace] = []
    query_index: Dict[str, List[int]] = {}
    query_traces: Dict[str, List[str]] = {}  # str(query_idx) → [trace_ids]
    dataset_rows: List[Dict[str, Any]] = []
    evaluation_job_id: Optional[str] = None
    status: str = "collecting"
