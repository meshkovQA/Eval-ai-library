# metric_pattern.py
"""
Base classes for evaluation metrics with beautiful console logging.
"""
import json
import time
from typing import Type, Dict, Any, Union, Optional

from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.llm_client import chat_complete


class Colors:
    """ANSI color codes for beautiful console output"""
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

    def __init__(self, model: str, threshold: float, verbose: bool = False):
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

    def print_result(self, result: Dict[str, Any]):
        """
        Print evaluation result based on verbose setting.
        If verbose=False: simple dict print
        If verbose=True: beautiful formatted output with colors
        """
        if not self.verbose:
            print(result)
            return

        WIDTH = 220

        success = result.get('success', False)
        score = result.get('score', 0.0)
        reason = result.get('reason', 'N/A')
        cost = result.get('evaluation_cost', 0.0)
        evaluation_log = result.get('evaluation_log', None)

        status_icon = "✅" if success else "❌"
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "PASSED" if success else "FAILED"

        bar_length = 40
        filled = int(bar_length * score)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Центрирование названия метрики
        metric_name = result.get('name', self.name)
        formatted_name = f"📊 {metric_name}"
        padding = max(0, WIDTH - len(formatted_name))
        left_pad = padding // 2
        right_pad = padding - left_pad
        centered_name = " " * left_pad + formatted_name + " " * right_pad

        # Рамка заголовка (WIDTH + 2 для границ)
        border = "═" * WIDTH

        print(f"""
    {Colors.BOLD}{Colors.CYAN}╔{border}╗{Colors.ENDC}
    {Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}{centered_name}{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}
    {Colors.BOLD}{Colors.CYAN}╚{border}╝{Colors.ENDC}

    {Colors.BOLD}Status:{Colors.ENDC}     {status_icon} {status_color}{Colors.BOLD}{status_text}{Colors.ENDC}

    {Colors.BOLD}Score:{Colors.ENDC}      {Colors.YELLOW}{score:.2f}{Colors.ENDC} [{bar}] {score*100:.0f}%

    {Colors.BOLD}Cost:{Colors.ENDC}       {Colors.BLUE}💰 ${cost:.6f}{Colors.ENDC}

    {Colors.BOLD}Reason:{Colors.ENDC}     {Colors.DIM}{reason}{Colors.ENDC}
    """)

        if evaluation_log:
            import json
            print(f"{Colors.BOLD}Evaluation Log:{Colors.ENDC}")
            log_border = "─" * WIDTH
            print(f"{Colors.DIM}╭{log_border}╮{Colors.ENDC}")

            log_json = json.dumps(evaluation_log, indent=4, ensure_ascii=False)

            for line in log_json.split('\n'):
                print(f"{Colors.DIM}│{Colors.ENDC} {line}")

            print(f"{Colors.DIM}╰{log_border}╯{Colors.ENDC}")

        print(f"\n{Colors.DIM}{'─'*WIDTH}{Colors.ENDC}\n")


class ConversationalMetricPattern:
    """
    Base class for conversational metrics (evaluating full dialogues).
    Used for metrics like RoleAdherence, DialogueCoherence, etc.
    """
    name: str

    def __init__(self, model: str, threshold: float, verbose: bool = False):
        self.model = model
        self.threshold = threshold
        self.verbose = verbose
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

    def print_result(self, result: Dict[str, Any]):
        """
        Print evaluation result based on verbose setting.
        If verbose=False: simple dict print
        If verbose=True: beautiful formatted output with colors
        """
        if not self.verbose:
            print(result)
            return

        WIDTH = 120

        success = result.get('success', False)
        score = result.get('score', 0.0)
        reason = result.get('reason', 'N/A')
        cost = result.get('evaluation_cost', 0.0)
        evaluation_log = result.get('evaluation_log', None)

        status_icon = "✅" if success else "❌"
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "PASSED" if success else "FAILED"

        bar_length = 40
        filled = int(bar_length * score)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Центрирование названия метрики
        metric_name = result.get('name', self.name)
        formatted_name = f"📊 {metric_name}"
        padding = max(0, WIDTH - len(formatted_name))
        left_pad = padding // 2
        right_pad = padding - left_pad
        centered_name = " " * left_pad + formatted_name + " " * right_pad

        # Рамка заголовка (WIDTH + 2 для границ)
        border = "═" * WIDTH

        print(f"""
    {Colors.BOLD}{Colors.CYAN}╔{border}╗{Colors.ENDC}
    {Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}{centered_name}{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}
    {Colors.BOLD}{Colors.CYAN}╚{border}╝{Colors.ENDC}

    {Colors.BOLD}Status:{Colors.ENDC}     {status_icon} {status_color}{Colors.BOLD}{status_text}{Colors.ENDC}

    {Colors.BOLD}Score:{Colors.ENDC}      {Colors.YELLOW}{score:.2f}{Colors.ENDC} [{bar}] {score*100:.0f}%

    {Colors.BOLD}Cost:{Colors.ENDC}       {Colors.BLUE}💰 ${cost:.6f}{Colors.ENDC}

    {Colors.BOLD}Reason:{Colors.ENDC}     {Colors.DIM}{reason}{Colors.ENDC}
    """)

        if evaluation_log:
            import json
            print(f"{Colors.BOLD}Evaluation Log:{Colors.ENDC}")
            log_border = "─" * WIDTH
            print(f"{Colors.DIM}╭{log_border}╮{Colors.ENDC}")

            log_json = json.dumps(evaluation_log, indent=4, ensure_ascii=False)

            for line in log_json.split('\n'):
                print(f"{Colors.DIM}│{Colors.ENDC} {line}")

            print(f"{Colors.DIM}╰{log_border}╯{Colors.ENDC}")

        print(f"\n{Colors.DIM}{'─'*WIDTH}{Colors.ENDC}\n")
