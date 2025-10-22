# metric_pattern.py
"""
Base classes for evaluation metrics with beautiful console logging.
"""
import json
import time
from typing import Type, Dict, Any, Union, Optional

from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.llm_client import chat_complete


# ANSI color codes for beautiful console output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


class MetricPattern:
    """
    Base class for metrics that use a pattern-based approach to evaluation.
    This class is designed to be subclassed for specific metrics.
    """
    name: str  # name of the metric

    def __init__(self, model: str, threshold: float, verbose: bool = True):
        self.model = model
        self.threshold = threshold
        self.verbose = verbose

    def _log(self, message: str, color: str = Colors.CYAN):
        """Log message with color if verbose mode is enabled"""
        if self.verbose:
            print(f"{color}{message}{Colors.ENDC}")

    def _log_step(self, step_name: str, step_num: int = None):
        """Log evaluation step"""
        if self.verbose:
            prefix = f"[{step_num}] " if step_num else ""
            print(f"{Colors.DIM}  {prefix}{step_name}...{Colors.ENDC}")

    async def evaluate(self, test_case: Union[EvalTestCase]) -> Dict[str, Any]:
        """
        Base evaluation method - override in subclasses for custom behavior.
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.BLUE}🔍 Evaluating: {self.name}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
            print(f"{Colors.DIM}Model: {self.model}{Colors.ENDC}")
            print(f"{Colors.DIM}Threshold: {self.threshold}{Colors.ENDC}")

        self._log_step("Generating evaluation prompt", 1)

        # 1) Generate prompt
        prompt = self.template.generate_prompt(
            test_case=test_case,
            threshold=self.threshold
        )

        self._log_step("Calling LLM", 2)

        # 2) Make API call
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        self._log_step("Parsing response", 3)

        # 3) Parse the response
        try:
            data = json.loads(text)
        except Exception as e:
            self._log(f"❌ Failed to parse JSON: {e}", Colors.RED)
            raise RuntimeError(
                f"Cannot parse JSON from model response: {e}\n{text}")

        score = float(data.get("score", 0.0))
        reason = data.get("reason")
        success = score >= self.threshold

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Log results
        if self.verbose:
            print(f"\n{Colors.BOLD}📊 Results:{Colors.ENDC}")
            score_color = Colors.GREEN if success else Colors.RED
            success_icon = "✅" if success else "❌"
            print(
                f"  {success_icon} Status: {score_color}{Colors.BOLD}{'PASSED' if success else 'FAILED'}{Colors.ENDC}")
            print(
                f"  📈 Score: {score_color}{score:.2f}{Colors.ENDC} (threshold: {self.threshold})")
            print(f"  💰 Cost: {Colors.YELLOW}${cost:.6f}{Colors.ENDC}")
            print(f"  ⏱️  Time: {Colors.DIM}{elapsed_time:.2f}s{Colors.ENDC}")
            if reason:
                print(
                    f"  💬 Reason: {Colors.DIM}{reason[:100]}{'...' if len(reason) > 100 else ''}{Colors.ENDC}")

        return {
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": cost,
        }


class ConversationalMetricPattern:
    """
    Base class for conversational metrics (evaluating full dialogues).
    Used for metrics like RoleAdherence, DialogueCoherence, etc.
    """
    name: str
    template_cls: Type

    def __init__(self, model: str, threshold: float, verbose: bool = True):
        self.model = model
        self.threshold = threshold
        self.verbose = verbose
        if self.template_cls:
            self.template = self.template_cls()
        else:
            self.template = None
        self.chatbot_role: Optional[str] = None

    def _log(self, message: str, color: str = Colors.CYAN):
        """Log message with color if verbose mode is enabled"""
        if self.verbose:
            print(f"{color}{message}{Colors.ENDC}")

    def _log_step(self, step_name: str, step_num: int = None):
        """Log evaluation step"""
        if self.verbose:
            prefix = f"[{step_num}] " if step_num else ""
            print(f"{Colors.DIM}  {prefix}{step_name}...{Colors.ENDC}")

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        """
        Evaluate conversational test case with logging.
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
            print(
                f"{Colors.BOLD}{Colors.BLUE}💬 Evaluating Conversation: {self.name}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
            print(f"{Colors.DIM}Model: {self.model}{Colors.ENDC}")
            print(f"{Colors.DIM}Threshold: {self.threshold}{Colors.ENDC}")
            print(f"{Colors.DIM}Turns: {len(test_case.turns)}{Colors.ENDC}")

        self._log_step("Generating evaluation prompt", 1)

        # 1. Generate prompt
        if hasattr(self.template, "generate_prompt"):
            try:
                prompt = self.template.generate_prompt(
                    test_case=test_case,
                    threshold=self.threshold,
                    chatbot_role=self.chatbot_role
                )
            except TypeError:
                prompt = self.template.generate_prompt(
                    test_case=test_case,
                    threshold=self.threshold,
                    temperature=0.0
                )
        else:
            raise RuntimeError("Template is missing method generate_prompt")

        self._log_step("Calling LLM", 2)

        # 2. Call API
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        self._log_step("Parsing response", 3)

        # 3. Parse response
        try:
            data = json.loads(text)
        except Exception as e:
            self._log(f"❌ Failed to parse JSON: {e}", Colors.RED)
            raise RuntimeError(
                f"Cannot parse JSON from model response: {e}\n{text}")

        score = float(data.get("score", 0.0))
        reason = data.get("reason")
        success = score >= self.threshold

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Log results
        if self.verbose:
            print(f"\n{Colors.BOLD}📊 Results:{Colors.ENDC}")
            score_color = Colors.GREEN if success else Colors.RED
            success_icon = "✅" if success else "❌"
            print(
                f"  {success_icon} Status: {score_color}{Colors.BOLD}{'PASSED' if success else 'FAILED'}{Colors.ENDC}")
            print(
                f"  📈 Score: {score_color}{score:.2f}{Colors.ENDC} (threshold: {self.threshold})")
            print(f"  💰 Cost: {Colors.YELLOW}${cost:.6f}{Colors.ENDC}")
            print(f"  ⏱️  Time: {Colors.DIM}{elapsed_time:.2f}s{Colors.ENDC}")
            if reason:
                print(
                    f"  💬 Reason: {Colors.DIM}{reason[:100]}{'...' if len(reason) > 100 else ''}{Colors.ENDC}")

        return {
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_cost": cost,
        }
