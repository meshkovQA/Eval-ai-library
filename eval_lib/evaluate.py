from dataclasses import asdict
import json
from typing import List, Tuple, Dict, Any
from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.metric_pattern import MetricPattern, ConversationalMetricPattern
from eval_lib.evaluation_schema import TestCaseResult, MetricResult, ConversationalTestCaseResult


async def evaluate(
    test_cases: List[EvalTestCase],
    metrics: List[MetricPattern],
) -> List[Tuple[None, List[TestCaseResult]]]:

    results: List[Tuple[None, List[TestCaseResult]]] = []

    for tc in test_cases:
        mdata = []
        for m in metrics:
            res = await m.evaluate(tc)
            # gathering the results
            mdata.append(MetricResult(
                name=m.name,
                score=res["score"],
                threshold=m.threshold,
                success=res["success"],
                evaluation_cost=res["evaluation_cost"],
                reason=res["reason"],
                evaluation_model=m.model,
                evaluation_log=res.get("evaluation_log", None)
            ))
        overall = all(d.success for d in mdata)
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

    print("\n=== EVALUATION RESULT ===")
    for meta, tc_list in results:

        print(f"Tuple meta part: {meta!r}")
        for tc in tc_list:

            tc_dict = asdict(tc)
            print(json.dumps(tc_dict, indent=2, ensure_ascii=False))
            print("-" * 50)

    return results


async def evaluate_conversations(
    conv_cases: List[ConversationalEvalTestCase],
    metrics: List[ConversationalMetricPattern],
) -> List[Tuple[None, List[ConversationalTestCaseResult]]]:

    results: List[Tuple[None, List[ConversationalTestCaseResult]]] = []

    for dlg in conv_cases:
        metric_rows: List[MetricResult] = []

        for m in metrics:
            res: Dict[str, Any] = await m.evaluate(dlg)

            metric_rows.append(
                MetricResult(
                    name=m.name,
                    score=res["score"],
                    threshold=m.threshold,
                    success=res["success"],
                    evaluation_cost=res.get("evaluation_cost"),
                    reason=res.get("reason"),
                    evaluation_model=m.model,
                    evaluation_log=res.get("evaluation_log"),
                )
            )

        overall_ok = all(r.success for r in metric_rows)

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

    print("\n=== CONVERSATIONAL EVALUATION RESULT ===")
    for _, conv_list in results:
        for conv in conv_list:
            print(json.dumps(asdict(conv), indent=2, ensure_ascii=False))
            print("-" * 70)

    return results
