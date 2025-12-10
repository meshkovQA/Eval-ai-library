import asyncio
import aiohttp
import json
from threading import Lock
from typing import List, Dict, Any
from .types import TraceSpan
from .config import TracingConfig


def _safe_serialize(obj: Any, seen: set = None) -> Any:
    """Recursively serialize an object to JSON-safe types"""
    if seen is None:
        seen = set()

    # Handle None and primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Prevent infinite recursion
    obj_id = id(obj)
    if obj_id in seen:
        return f"<circular ref: {type(obj).__name__}>"
    seen.add(obj_id)

    try:
        # Handle UUID
        if hasattr(obj, 'hex'):
            return str(obj)

        # Handle dict
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v, seen) for k, v in obj.items()}

        # Handle list/tuple
        if isinstance(obj, (list, tuple)):
            return [_safe_serialize(item, seen) for item in obj]

        # Handle Pydantic models (v1 and v2)
        if hasattr(obj, 'model_dump'):
            try:
                return _safe_serialize(obj.model_dump(), seen)
            except Exception:
                pass
        if hasattr(obj, 'dict') and callable(obj.dict):
            try:
                return _safe_serialize(obj.dict(), seen)
            except Exception:
                pass

        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            try:
                from dataclasses import asdict
                return _safe_serialize(asdict(obj), seen)
            except Exception:
                pass

        # Handle LangChain objects (AgentAction, ToolAgentAction, etc.)
        if hasattr(obj, 'to_dict'):
            try:
                return _safe_serialize(obj.to_dict(), seen)
            except Exception:
                pass

        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            try:
                result = {"_type": type(obj).__name__}
                for k, v in obj.__dict__.items():
                    if not k.startswith('_'):
                        result[k] = _safe_serialize(v, seen)
                return result
            except Exception:
                pass

        # Handle iterables
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return [_safe_serialize(item, seen) for item in obj]
            except Exception:
                pass

        # Fallback to string representation
        return str(obj)
    finally:
        seen.discard(obj_id)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects gracefully"""

    def default(self, obj: Any) -> Any:
        return _safe_serialize(obj)


class TraceSender:
    """Sender for trace spans - groups spans by trace_id"""

    def __init__(self):
        self._lock = Lock()
        # Store spans grouped by trace_id
        self._traces: Dict[str, List[TraceSpan]] = {}
        # Store trace-level metadata (model, tokens, output)
        self._trace_metadata: Dict[str, Dict[str, Any]] = {}

    def set_trace_metadata(self, trace_id: str, metadata: Dict[str, Any]):
        """Set trace-level metadata (model, tokens, final output)"""
        with self._lock:
            if trace_id not in self._trace_metadata:
                self._trace_metadata[trace_id] = {}
            self._trace_metadata[trace_id].update(metadata)

    def add_span(self, span: TraceSpan):
        """Add a span to its trace group"""
        with self._lock:
            trace_id = span.trace_id
            if trace_id not in self._traces:
                self._traces[trace_id] = []
            self._traces[trace_id].append(span)

    def flush_trace(self, trace_id: str):
        """Send all spans for a specific trace"""
        with self._lock:
            if trace_id not in self._traces:
                return
            spans = self._traces.pop(trace_id)
            # Get and remove trace metadata
            trace_meta = self._trace_metadata.pop(trace_id, {})

        if not spans:
            return

        self._send_trace_sync(trace_id, spans, trace_meta)

    def _send_trace_sync(self, trace_id: str, spans: List[TraceSpan], trace_meta: Dict[str, Any] = None):
        """Synchronously send a complete trace"""
        url = TracingConfig.get_url()
        if not url:
            return

        # Build hierarchical trace structure
        trace_data = self._build_trace_structure(trace_id, spans, trace_meta or {})

        payload = {
            "project": TracingConfig.get_project(),
            "trace": trace_data
        }

        headers = {"Content-Type": "application/json"}
        api_key = TracingConfig.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context - schedule the coroutine
            asyncio.ensure_future(self._async_send(url, payload, headers))
        except RuntimeError:
            # No running loop - create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_send(url, payload, headers))
            finally:
                loop.close()

    def _build_trace_structure(self, trace_id: str, spans: List[TraceSpan], trace_meta: Dict[str, Any] = None) -> dict:
        """Build hierarchical trace structure from flat spans list"""

        # Create lookup by span_id
        span_map = {span.span_id: span.to_dict() for span in spans}

        # Find root spans (no parent or parent not in this trace)
        root_spans = []
        child_map: Dict[str, List[dict]] = {}

        for span in spans:
            span_dict = span_map[span.span_id]
            parent_id = span.parent_span_id

            if parent_id is None or parent_id not in span_map:
                root_spans.append(span_dict)
            else:
                if parent_id not in child_map:
                    child_map[parent_id] = []
                child_map[parent_id].append(span_dict)

        # Build tree recursively
        def attach_children(span_dict: dict):
            span_id = span_dict.get("span_id")
            if span_id in child_map:
                span_dict["children"] = child_map[span_id]
                for child in span_dict["children"]:
                    attach_children(child)

        for root in root_spans:
            attach_children(root)

        # Calculate trace-level metadata
        all_times = [s.start_time for s in spans if s.start_time]
        end_times = [s.end_time for s in spans if s.end_time]

        # Calculate response_time from root span's duration_ms if available
        response_time = None
        if root_spans:
            # Get duration from the first root span (main execution span)
            root_duration_ms = root_spans[0].get("duration_ms")
            if root_duration_ms:
                response_time = round(root_duration_ms / 1000, 3)  # Convert ms to seconds

        # Collect tools called during the trace (only top-level tool calls, not nested)
        # Build a set of span_ids that are tool_calls
        tool_span_ids = {span.span_id for span in spans if span.span_type and span.span_type.value == "tool_call"}

        tools_called = []
        for span in spans:
            if span.span_type and span.span_type.value == "tool_call":
                # Only include if parent is not a tool_call (top-level tool)
                if span.parent_span_id not in tool_span_ids:
                    tools_called.append(span.name)

        # Try to extract tokens from LLM spans if not in trace_meta
        # This is useful when streaming is enabled and tokens are not in callback
        extracted_input_tokens = 0
        extracted_output_tokens = 0
        for span in spans:
            if span.span_type and span.span_type.value == "llm_call" and span.output:
                # Try to get tokens from span output
                output = span.output
                if isinstance(output, dict):
                    llm_output = output.get("llm_output", {})
                    if llm_output:
                        token_usage = llm_output.get("token_usage", {})
                        if token_usage:
                            extracted_input_tokens += token_usage.get("prompt_tokens", 0)
                            extracted_output_tokens += token_usage.get("completion_tokens", 0)

        result = {
            "trace_id": trace_id,
            "start_time": min(all_times) if all_times else None,
            "end_time": max(end_times) if end_times else None,
            "response_time": response_time,
            "tools_called": tools_called if tools_called else None,
            "spans": root_spans,
            "span_count": len(spans)
        }

        # Add trace-level metadata (model, input, output, tokens, timing) at the top
        if trace_meta:
            # Insert metadata fields before spans
            if "model" in trace_meta:
                result["model"] = trace_meta["model"]
            if "input" in trace_meta:
                result["input"] = trace_meta["input"]
            if "output" in trace_meta:
                result["output"] = trace_meta["output"]
            if "input_tokens" in trace_meta:
                result["input_tokens"] = trace_meta["input_tokens"]
            if "output_tokens" in trace_meta:
                result["output_tokens"] = trace_meta["output_tokens"]
            if "total_tokens" in trace_meta:
                result["total_tokens"] = trace_meta["total_tokens"]
            if "response_time" in trace_meta:
                result["response_time"] = trace_meta["response_time"]
            # Add any other custom metadata
            for key, value in trace_meta.items():
                if key not in result:
                    result[key] = value

        return result

    async def _async_send(self, url: str, payload: dict, headers: dict):
        """Async HTTP POST"""
        try:
            # Use custom encoder to handle non-serializable objects
            json_data = json.dumps(payload, cls=SafeJSONEncoder)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=json_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    pass  # Silently handle response
        except Exception:
            pass  # Silently handle exceptions

    def flush(self):
        """Send all pending traces"""
        with self._lock:
            trace_ids = list(self._traces.keys())

        for trace_id in trace_ids:
            self.flush_trace(trace_id)

    def stop(self):
        """Stop the sender and flush remaining traces"""
        self.flush()