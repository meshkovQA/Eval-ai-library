from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"


class HeaderEntry(BaseModel):
    key: str
    value: str
    enabled: bool = True


class ApiConnectionConfig(BaseModel):
    name: str = "Untitled"
    base_url: str = ""
    method: HttpMethod = HttpMethod.POST
    headers: List[HeaderEntry] = []
    query_params: Dict[str, str] = {}
    body_template: str = ""
    timeout_seconds: int = 60
    max_retries: int = 1
    delay_between_requests_ms: int = 0


class ResponseMapping(BaseModel):
    actual_output_path: str = ""
    retrieval_context_path: Optional[str] = None
    tools_called_path: Optional[str] = None
    token_usage_path: Optional[str] = None
    system_prompt_path: Optional[str] = None


class DatasetColumnMapping(BaseModel):
    input_column: str = ""
    expected_output_column: Optional[str] = None
    context_column: Optional[str] = None
    tools_called_column: Optional[str] = None
    expected_tools_column: Optional[str] = None
    template_variable_map: Dict[str, str] = {}


class MetricConfig(BaseModel):
    metric_class: str
    params: Dict[str, Any] = {}


class EvalJobConfig(BaseModel):
    id: Optional[str] = None
    name: str = "Untitled Job"
    api_config: ApiConnectionConfig = ApiConnectionConfig()
    response_mapping: ResponseMapping = ResponseMapping()
    dataset_column_mapping: DatasetColumnMapping = DatasetColumnMapping()
    metrics: List[MetricConfig] = []
    eval_model: str = "gpt-4o-mini"
    cost_per_1m_tokens: float = 0.0
    created_at: Optional[str] = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    total_rows: int = 0
    completed_rows: int = 0
    current_phase: str = ""
    errors: List[str] = []
    result_session_id: Optional[str] = None
    response_times_ms: List[int] = []
    avg_response_time_ms: int = 0
