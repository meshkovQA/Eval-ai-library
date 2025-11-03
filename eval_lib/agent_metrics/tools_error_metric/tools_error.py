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

    def _extract_tool_history(self, test_case: EvalTestCase) -> str:
        """Extract tool call history from test case"""

        # Try to get tool history from various possible locations
        tool_history = None

        # Check if there's a tool_calls field
        if hasattr(test_case, 'tool_calls'):
            tool_history = test_case.tool_calls
        # Check context
        elif hasattr(test_case, 'context') and test_case.context:
            if isinstance(test_case.context, dict) and 'tool_calls' in test_case.context:
                tool_history = test_case.context['tool_calls']
            elif isinstance(test_case.context, str) and 'tool' in test_case.context.lower():
                tool_history = test_case.context
        # Check expected_output if it contains tool info
        elif hasattr(test_case, 'expected_output') and test_case.expected_output:
            if isinstance(test_case.expected_output, str) and 'tool' in test_case.expected_output.lower():
                tool_history = test_case.expected_output

        if tool_history is None:
            return "No tool calls were made"

        # Format tool history if it's a list or dict
        if isinstance(tool_history, list):
            formatted = []
            for i, call in enumerate(tool_history, 1):
                if isinstance(call, dict):
                    func_name = call.get(
                        'function', call.get('name', 'unknown'))
                    params = call.get('parameters', call.get('args', {}))
                    result = call.get('result', call.get('output', 'N/A'))
                    formatted.append(f"{i}. {func_name}({params}) -> {result}")
                else:
                    formatted.append(f"{i}. {call}")
            return "\n".join(formatted)
        elif isinstance(tool_history, dict):
            formatted = []
            for i, (call_id, call_data) in enumerate(tool_history.items(), 1):
                formatted.append(f"{i}. {call_data}")
            return "\n".join(formatted)
        else:
            return str(tool_history)

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
