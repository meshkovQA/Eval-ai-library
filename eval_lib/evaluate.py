# evaluate.py
"""
Main evaluation functions with beautiful console progress tracking.
"""
from dataclasses import asdict
import json
import time
from typing import List, Tuple, Dict, Any
from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.metric_pattern import MetricPattern, ConversationalMetricPattern
from eval_lib.evaluation_schema import TestCaseResult, MetricResult, ConversationalTestCaseResult


# ANSI color codes
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


def _print_header(title: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def _print_progress(current: int, total: int, item_name: str):
    """Print progress bar"""
    percentage = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(
        f"\r{Colors.CYAN}Progress: [{bar}] {percentage:.0f}% ({current}/{total}) - {item_name}{Colors.ENDC}", end='', flush=True)


def _print_summary(results: List, total_cost: float, total_time: float, passed: int, total: int):
    """Print evaluation summary"""
    print(f"\n\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}📋 EVALUATION SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")

    success_rate = (passed / total * 100) if total > 0 else 0
    status_color = Colors.GREEN if success_rate >= 80 else Colors.YELLOW if success_rate >= 50 else Colors.RED

    print(f"\n{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  ✅ Passed: {Colors.GREEN}{passed}{Colors.ENDC} / {total}")
    print(f"  ❌ Failed: {Colors.RED}{total - passed}{Colors.ENDC} / {total}")
    print(f"  📊 Success Rate: {status_color}{success_rate:.1f}%{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Resource Usage:{Colors.ENDC}")
    print(f"  💰 Total Cost: {Colors.YELLOW}${total_cost:.6f}{Colors.ENDC}")
    print(f"  ⏱️  Total Time: {Colors.CYAN}{total_time:.2f}s{Colors.ENDC}")
    print(
        f"  📈 Avg Time per Test: {Colors.DIM}{(total_time/total if total > 0 else 0):.2f}s{Colors.ENDC}")

    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}\n")


async def evaluate(
    test_cases: List[EvalTestCase],
    metrics: List[MetricPattern],
    verbose: bool = True
) -> List[Tuple[None, List[TestCaseResult]]]:
    """
    Evaluate test cases with multiple metrics.

    Args:
        test_cases: List of test cases to evaluate
        metrics: List of metrics to apply
        verbose: Enable detailed logging (default: True)

    Returns:
        List of evaluation results
    """
    start_time = time.time()
    results: List[Tuple[None, List[TestCaseResult]]] = []

    total_cost = 0.0
    total_passed = 0
    total_tests = len(test_cases)

    if verbose:
        _print_header("🚀 STARTING EVALUATION")
        print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
        print(f"  📝 Test Cases: {Colors.CYAN}{total_tests}{Colors.ENDC}")
        print(f"  📊 Metrics: {Colors.CYAN}{len(metrics)}{Colors.ENDC}")
        print(
            f"  🎯 Total Evaluations: {Colors.CYAN}{total_tests * len(metrics)}{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Metrics:{Colors.ENDC}")
        for i, m in enumerate(metrics, 1):
            print(
                f"  {i}. {Colors.BLUE}{m.name}{Colors.ENDC} (threshold: {m.threshold})")

    # Process each test case
    for tc_idx, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.ENDC}")
            print(
                f"{Colors.BOLD}{Colors.CYAN}📝 Test Case {tc_idx}/{total_tests}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.ENDC}")
            print(
                f"{Colors.DIM}Input: {tc.input[:80]}{'...' if len(tc.input) > 80 else ''}{Colors.ENDC}")

        mdata = []
        test_cost = 0.0

        # Evaluate with each metric
        for m_idx, m in enumerate(metrics, 1):
            if verbose:
                _print_progress(m_idx, len(metrics), m.name)

            # Set verbose flag for metrics
            original_verbose = getattr(m, 'verbose', True)
            m.verbose = verbose

            res = await m.evaluate(tc)

            # Restore original verbose setting
            m.verbose = original_verbose

            # Gather results
            cost = res.get("evaluation_cost", 0) or 0
            test_cost += cost
            total_cost += cost

            mdata.append(MetricResult(
                name=m.name,
                score=res["score"],
                threshold=m.threshold,
                success=res["success"],
                evaluation_cost=cost,
                reason=res["reason"],
                evaluation_model=m.model,
                evaluation_log=res.get("evaluation_log", None)
            ))

        overall = all(d.success for d in mdata)
        if overall:
            total_passed += 1

        if verbose:
            print(f"\n{Colors.BOLD}Test Case Summary:{Colors.ENDC}")
            tc_status_color = Colors.GREEN if overall else Colors.RED
            tc_status_icon = "✅" if overall else "❌"
            print(
                f"  {tc_status_icon} Overall: {tc_status_color}{Colors.BOLD}{'PASSED' if overall else 'FAILED'}{Colors.ENDC}")
            print(f"  💰 Cost: {Colors.YELLOW}${test_cost:.6f}{Colors.ENDC}")

            # Show metric breakdown
            print(f"\n  {Colors.BOLD}Metrics Breakdown:{Colors.ENDC}")
            for md in mdata:
                status = "✅" if md.success else "❌"
                color = Colors.GREEN if md.success else Colors.RED
                print(
                    f"    {status} {md.name}: {color}{md.score:.2f}{Colors.ENDC}")

        results.append((None, [TestCaseResult(
            input=tc.input,
            actual_output=tc.actual_output,
            expected_output=tc.expected_output,
            retrieval_context=tc.retrieval_context,
            tools_called=tc.tools_called,
            expected_tools=tc.expected_tools,
            success=overall,
            metrics_data=mdata
        )]))

    # Calculate total time
    total_time = time.time() - start_time

    # Print summary
    if verbose:
        _print_summary(results, total_cost, total_time,
                       total_passed, total_tests)

    return results


async def evaluate_conversations(
    conv_cases: List[ConversationalEvalTestCase],
    metrics: List[ConversationalMetricPattern],
    verbose: bool = True
) -> List[Tuple[None, List[ConversationalTestCaseResult]]]:
    """
    Evaluate conversational test cases with multiple metrics.

    Args:
        conv_cases: List of conversational test cases
        metrics: List of conversational metrics
        verbose: Enable detailed logging (default: True)

    Returns:
        List of evaluation results
    """
    start_time = time.time()
    results: List[Tuple[None, List[ConversationalTestCaseResult]]] = []

    total_cost = 0.0
    total_passed = 0
    total_conversations = len(conv_cases)

    if verbose:
        _print_header("🚀 STARTING CONVERSATIONAL EVALUATION")
        print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
        print(
            f"  💬 Conversations: {Colors.CYAN}{total_conversations}{Colors.ENDC}")
        print(f"  📊 Metrics: {Colors.CYAN}{len(metrics)}{Colors.ENDC}")
        print(
            f"  🎯 Total Evaluations: {Colors.CYAN}{total_conversations * len(metrics)}{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Metrics:{Colors.ENDC}")
        for i, m in enumerate(metrics, 1):
            print(
                f"  {i}. {Colors.BLUE}{m.name}{Colors.ENDC} (threshold: {m.threshold})")

    # Process each conversation
    for conv_idx, dlg in enumerate(conv_cases, 1):
        if verbose:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.ENDC}")
            print(
                f"{Colors.BOLD}{Colors.CYAN}💬 Conversation {conv_idx}/{total_conversations}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.ENDC}")
            print(f"{Colors.DIM}Turns: {len(dlg.turns)}{Colors.ENDC}")
            if dlg.chatbot_role:
                print(
                    f"{Colors.DIM}Role: {dlg.chatbot_role[:60]}{'...' if len(dlg.chatbot_role) > 60 else ''}{Colors.ENDC}")

        metric_rows: List[MetricResult] = []
        conv_cost = 0.0

        # Evaluate with each metric
        for m_idx, m in enumerate(metrics, 1):
            if verbose:
                _print_progress(m_idx, len(metrics), m.name)

            # Set verbose flag for metrics
            original_verbose = getattr(m, 'verbose', True)
            m.verbose = verbose

            res: Dict[str, Any] = await m.evaluate(dlg)

            # Restore original verbose setting
            m.verbose = original_verbose

            cost = res.get("evaluation_cost", 0) or 0
            conv_cost += cost
            total_cost += cost

            metric_rows.append(
                MetricResult(
                    name=m.name,
                    score=res["score"],
                    threshold=m.threshold,
                    success=res["success"],
                    evaluation_cost=cost,
                    reason=res.get("reason"),
                    evaluation_model=m.model,
                    evaluation_log=res.get("evaluation_log"),
                )
            )

        overall_ok = all(r.success for r in metric_rows)
        if overall_ok:
            total_passed += 1

        if verbose:
            print(f"\n{Colors.BOLD}Conversation Summary:{Colors.ENDC}")
            conv_status_color = Colors.GREEN if overall_ok else Colors.RED
            conv_status_icon = "✅" if overall_ok else "❌"
            print(
                f"  {conv_status_icon} Overall: {conv_status_color}{Colors.BOLD}{'PASSED' if overall_ok else 'FAILED'}{Colors.ENDC}")
            print(f"  💰 Cost: {Colors.YELLOW}${conv_cost:.6f}{Colors.ENDC}")

            # Show metric breakdown
            print(f"\n  {Colors.BOLD}Metrics Breakdown:{Colors.ENDC}")
            for mr in metric_rows:
                status = "✅" if mr.success else "❌"
                color = Colors.GREEN if mr.success else Colors.RED
                print(
                    f"    {status} {mr.name}: {color}{mr.score:.2f}{Colors.ENDC}")

        dialogue_raw = []
        for turn in dlg.turns:
            dialogue_raw.append({"role": "user", "content": turn.input})
            dialogue_raw.append(
                {"role": "assistant", "content": turn.actual_output})

        conv_res = ConversationalTestCaseResult(
            dialogue=dialogue_raw,
            success=overall_ok,
            metrics_data=metric_rows,
        )
        results.append((None, [conv_res]))

    # Calculate total time
    total_time = time.time() - start_time

    # Print summary
    if verbose:
        _print_summary(results, total_cost, total_time,
                       total_passed, total_conversations)

    return results
