# tools_error.py
"""
Tools Error Detection Metric: Detects errors in AI agent's tool/function usage

This metric identifies whether an AI agent made errors when using tools or functions,
including:
- Incorrect function parameters
- Calling non-existent functions
- Improper call sequencing
- Ignoring tool results
- Repeated failed attempts
- Improper error handling

The metric analyzes the tool call history and results to detect usage errors.

Score range: 0.0 (no errors detected) to 1.0 (errors detected with high confidence)
"""
import json
from typing import Dict, Any, List, Optional
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.llm_client import chat_complete
from eval_lib.utils import extract_json_block
from eval_lib.tracing.spans import top_level_tool_calls


class ToolsErrorMetric(MetricPattern):
    name = "toolsError"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        error_types: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize Tools Error Detection Metric

        Args:
            model: LLM model to use for evaluation
            threshold: Detection threshold (0.0-1.0). Score >= threshold means errors detected
            error_types: List of error types to detect. If None, uses all types.
                        Types: parameter_error, invalid_function, sequence_error, 
                        result_ignored, repeated_failure, error_handling
            verbose: Enable verbose logging
        """
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.error_types = error_types or [
            "parameter_error", "invalid_function", "sequence_error",
            "result_ignored", "repeated_failure", "error_handling"
        ]

    # ==================== PROMPTS ====================

    @staticmethod
    def _few_shots() -> str:
        """Few-shot examples for tools error detection"""
        return """### EXAMPLE 1
Tool Calls History:
1. get_weather(city="New York") -> {"temp": 72, "condition": "sunny"}
2. [AI responds: "The weather in New York is 72°F and sunny"]

JSON:
{"detected": false, "confidence": 0.95, "reason": "Tool was called correctly with valid parameters and result was properly used", 
"error_types": [], "errors": [], "error_severity": "none"}

### EXAMPLE 2
Tool Calls History:
1. calculate_sum(a=5, b="hello") -> ERROR: TypeError - parameter 'b' must be numeric
2. calculate_sum(a=5, b="world") -> ERROR: TypeError - parameter 'b' must be numeric
3. [AI responds: "I couldn't calculate the sum"]

JSON:
{"detected": true, "confidence": 0.92, "reason": "Agent repeatedly passed string to numeric parameter without correction", 
"error_types": ["parameter_error", "repeated_failure"], 
"errors": [
    {"type": "parameter_error", "description": "Passing string 'hello' to numeric parameter 'b'"},
    {"type": "repeated_failure", "description": "Made same parameter error twice without learning"}
], 
"error_severity": "medium"}

### EXAMPLE 3
Tool Calls History:
1. search_database(query="user data") -> {"results": [{"id": 1, "name": "John"}]}
2. delete_record(id=5) -> ERROR: Record not found
3. [AI responds: "I deleted John's record"]

JSON:
{"detected": true, "confidence": 0.88, "reason": "Agent ignored tool error and claimed success incorrectly", 
"error_types": ["result_ignored", "error_handling"], 
"errors": [
    {"type": "result_ignored", "description": "Ignored error from delete_record"},
    {"type": "error_handling", "description": "Claimed success despite tool failure"}
], 
"error_severity": "high"}

### EXAMPLE 4
Tool Calls History:
1. get_user_permissions(user_id=123) -> {"can_delete": false}
2. delete_file(file_id=456, user_id=123) -> ERROR: Permission denied
3. [AI responds: "I cannot delete this file as you don't have the necessary permissions"]

JSON:
{"detected": false, "confidence": 0.93, "reason": "Agent correctly handled permission check and error, providing appropriate response", 
"error_types": [], "errors": [], "error_severity": "none"}

### EXAMPLE 5
Tool Calls History:
1. send_email_notification() -> ERROR: Missing required parameter 'recipient'
2. [AI responds: "Email sent successfully!"]

JSON:
{"detected": true, "confidence": 0.96, "reason": "Agent claimed success despite tool error due to missing parameter", 
"error_types": ["parameter_error", "result_ignored"], 
"errors": [
    {"type": "parameter_error", "description": "Missing required parameter 'recipient'"},
    {"type": "result_ignored", "description": "Ignored error and falsely claimed success"}
], 
"error_severity": "high"}

### EXAMPLE 6
Tool Calls History:
1. create_user(name="Alice") -> ERROR: Missing required parameter 'email'
2. create_user(name="Alice", email="alice@example.com") -> {"user_id": 789, "status": "created"}
3. [AI responds: "I created the user Alice with ID 789"]

JSON:
{"detected": false, "confidence": 0.90, "reason": "Agent made initial parameter error but corrected it in second attempt and used result correctly", 
"error_types": [], "errors": [], "error_severity": "none"}"""

    @classmethod
    def _prompt_evaluate(cls, test_case: EvalTestCase, tool_history: str, error_types: List[str]) -> str:
        """Generate evaluation prompt for tools error detection"""

        error_types_str = ", ".join(error_types)

        return f"""You are an AI agent evaluation expert analyzing tool/function usage for errors.

**Error types to detect:**
1. **parameter_error**: Wrong parameter types, missing required parameters, invalid values
2. **invalid_function**: Calling functions that don't exist or aren't available
3. **sequence_error**: Calling functions in wrong order (e.g., update before create)
4. **result_ignored**: Ignoring tool results or errors when making decisions
5. **repeated_failure**: Making the same error multiple times without correction
6. **error_handling**: Poor error handling, claiming success despite failures

**Important distinctions:**
- Single corrected mistake vs. repeated errors
- Proper error acknowledgment vs. ignoring errors
- Valid parameter passing vs. type mismatches
- Appropriate tool selection vs. invalid calls

Analyze the tool usage history and determine if errors occurred.

**Detection focus:** {error_types_str}

Return ONLY valid JSON:
{{
    "detected": <boolean, true if errors found>,
    "confidence": <float 0.0-1.0, confidence in detection>,
    "reason": <string explaining the detection>,
    "error_types": [<list of error types detected>],
    "errors": [
        {{
            "type": <error type>,
            "description": <specific error description>
        }}
    ],
    "error_severity": <"none"|"low"|"medium"|"high">
}}

---
{cls._few_shots()}
---
USER INPUT:
{test_case.input}

TOOL CALLS HISTORY:
{tool_history}

AI FINAL RESPONSE:
{test_case.actual_output}

JSON:"""

    # ==================== CORE EVALUATION ====================

    # ---- tool-history rendering ------------------------------------------

    @staticmethod
    def _get(step: Any, key: str, default: Any = None) -> Any:
        """Field access for a ``TraceStep`` model or a plain dict."""
        if isinstance(step, dict):
            return step.get(key, default)
        return getattr(step, key, default)

    @staticmethod
    def _render_value(value: Any) -> str:
        """Compact, JSON-shaped rendering of a tool output."""
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _render_args(cls, value: Any) -> str:
        """``key=value`` argument list, matching the few-shot examples."""
        if value is None:
            return ""
        if isinstance(value, dict):
            return ", ".join(
                f"{k}={json.dumps(v, ensure_ascii=False, default=str)}"
                for k, v in value.items()
            )
        if isinstance(value, str):
            return value
        return cls._render_value(value)

    @classmethod
    def _render_steps(cls, steps: List[Any]) -> str:
        """``N. name(args) -> output`` lines; failed calls show ``ERROR: …``."""
        lines = []
        for i, step in enumerate(steps, 1):
            name = cls._get(step, "name") or "unknown_tool"
            args = cls._render_args(cls._get(step, "input"))
            status = str(cls._get(step, "status") or "").lower()
            error = cls._get(step, "error")
            output = cls._get(step, "output")
            if status == "error" or error:
                detail = error or output or "unknown error"
                error_type = cls._get(step, "error_type")
                prefix = f"ERROR: {error_type} - " if error_type and error_type != "Exception" else "ERROR: "
                result = f"{prefix}{cls._render_value(detail)}"
            else:
                result = cls._render_value(output)
            lines.append(f"{i}. {name}({args}) -> {result}")
        return "\n".join(lines)

    @classmethod
    def _render_legacy(cls, tool_history: Any) -> str:
        """Render the free-form ``tool_calls`` shapes accepted via ``extra_fields``."""
        if isinstance(tool_history, list):
            formatted = []
            for i, call in enumerate(tool_history, 1):
                if isinstance(call, dict):
                    func_name = call.get("function", call.get("name", "unknown"))
                    params = call.get("parameters", call.get("args", call.get("input", {})))
                    result = call.get("result", call.get("output", "N/A"))
                    formatted.append(f"{i}. {func_name}({cls._render_args(params)}) -> {cls._render_value(result)}")
                else:
                    formatted.append(f"{i}. {call}")
            return "\n".join(formatted)
        if isinstance(tool_history, dict):
            return "\n".join(
                f"{i}. {call_data}" for i, (_, call_data) in enumerate(tool_history.items(), 1)
            )
        return str(tool_history)

    def _extract_tool_history(self, test_case: EvalTestCase) -> str:
        """Extract tool call history from the test case.

        Source of truth is ``execution_trace`` — the spans collected by the
        tracing subsystem (online) or loaded from a trace file (offline).
        Only ``tool_call`` steps are rendered, in order, as
        ``name(args) -> output`` with ``ERROR: …`` for failed calls, which is
        the shape the few-shot examples teach the judge.

        Fallbacks, in order: ``extra_fields["tool_calls"]`` (free-form list /
        dict), then ``tools_called`` (names only — arguments and results are
        unknown, and the judge is told so).
        """
        steps = getattr(test_case, "execution_trace", None) or []
        tool_steps = [s for s in steps if str(self._get(s, "type") or "").lower() == "tool_call"]

        # One line per *invocation*. A tool step nested under another tool
        # step, or a same-named one whose time window lies inside another's,
        # is the same call recorded by a second instrumentation layer (a
        # decorated function inside a framework's tool step) — not a retry.
        # The rules are structural, never "same name and arguments": sibling
        # repeats run one after another and are all kept, because genuine
        # retries are exactly what `repeated_failure` has to see.
        def _end(step: Any) -> Optional[float]:
            start, duration = self._get(step, "timestamp"), self._get(step, "duration_ms")
            if start is None or duration is None:
                return None
            return float(start) + float(duration) / 1000.0

        tool_steps = top_level_tool_calls(
            tool_steps,
            get_id=lambda s: self._get(s, "step_id"),
            get_parent=lambda s: self._get(s, "parent_step_id"),
            get_name=lambda s: self._get(s, "name"),
            get_start=lambda s: self._get(s, "timestamp"),
            get_end=_end,
        )

        if tool_steps:
            # Keep chronological order when timestamps are present.
            if all(self._get(s, "timestamp") is not None for s in tool_steps):
                tool_steps = sorted(tool_steps, key=lambda s: self._get(s, "timestamp"))
            return self._render_steps(tool_steps)

        extra = getattr(test_case, "extra_fields", None) or {}
        legacy = extra.get("tool_calls") if isinstance(extra, dict) else None
        if legacy:
            return self._render_legacy(legacy)

        names = getattr(test_case, "tools_called", None) or []
        if names:
            lines = [f"{i}. {name}(...) -> (result not recorded)" for i, name in enumerate(names, 1)]
            lines.append("(Only tool names are available — arguments and results were not captured.)")
            return "\n".join(lines)

        return "No tool calls were made"

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        """
        Detect errors in AI agent's tool usage.

        Returns:
            Dictionary with:
            - score: Detection confidence (0.0-1.0)
            - success: True if errors detected with confidence >= threshold
            - reason: Explanation of detection
            - evaluation_cost: LLM evaluation cost
            - evaluation_log: Detailed analysis
        """
        total_cost = 0.0

        self._log("🔍 Detecting tools usage errors")

        # Step 1: Extract tool history
        tool_history = self._extract_tool_history(test_case)
        self._log_step(f"Analyzing tool call history", 1)

        # Step 2: Generate evaluation prompt
        prompt = self._prompt_evaluate(
            test_case, tool_history, self.error_types)

        # Step 3: Get LLM evaluation
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        total_cost += cost or 0.0

        # Step 4: Parse response
        try:
            extracted = extract_json_block(text)
            data = json.loads(extracted)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse JSON response: {e}\n{text}")

        # Step 5: Extract results
        detected = data.get("detected", False)
        confidence = data.get("confidence", 0.0)
        reason = data.get("reason", "")
        error_types = data.get("error_types", [])
        errors = data.get("errors", [])
        error_severity = data.get("error_severity", "unknown")

        # Step 6: Determine success based on confidence
        score = confidence
        success = detected and score >= self.threshold

        # Step 7: Build evaluation log
        evaluation_log = {
            "user_input": test_case.input,
            "comment_user_input": "The user input that triggered the AI agent's tool usage.",
            "tool_history": tool_history,
            "comment_tool_history": "The history of tool calls made by the AI agent.",
            "ai_response": test_case.actual_output,
            "comment_ai_response": "The AI agent's final response after tool usage.",
            "error_types_filter": self.error_types,
            "comment_error_types_filter": "Types of tool errors being searched for.",
            "detected": detected,
            "comment_detected": "Whether tool usage errors were detected.",
            "confidence": confidence,
            "comment_confidence": "Confidence level of the detection (0.0-1.0).",
            "error_types": error_types,
            "comment_error_types": "Types of errors detected: parameter_error, invalid_function, sequence_error, etc.",
            "errors": errors,
            "comment_errors": "List of specific errors found with descriptions.",
            "error_severity": error_severity,
            "comment_error_severity": "Severity of the errors: none, low, medium, high.",
            "score": score,
            "comment_score": "Detection confidence score (0.0-1.0). Higher score means more confident error detection.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the detection confidence meets the required threshold."
        }

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log
        }

        self.print_result(result)
        return result
