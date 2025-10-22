import json
import re
from math import exp
from typing import List, Dict, Any, Tuple

from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.llm_client import chat_complete


VERDICT_WEIGHTS = {
    "fully":   1.0,
    "mostly":  0.8,
    "partial": 0.5,
    "minor":   0.2,
    "none":    0.0,
}

LABEL_HELP = """
Rate knowledge retention (worst → best):

none   – assistant contradicts or forgets previous facts  
minor  – one small lapse or omission  
partial– several lapses but overall context kept  
mostly – slight omissions, no contradiction  
fully  – remembers every relevant fact
""".strip()

FEW_SHOT = """
Example GOOD:
Conversation:
1. User: What year was Python created?
   Assistant: Python was first released in 1991.
2. User: Remind me, who created it?
   Assistant: It was created by Guido van Rossum in 1991.
Verdicts:
[{"verdict":"fully","reason":"Assistant repeated year and author correctly"}]

Example BAD:
Conversation:
1. User: I live in Spain.
   Assistant: Great! How's the weather?
2. User: Remind me later that I live in Spain.
   Assistant: Sure, I'll remind you that you live in Italy.
Verdicts:
[{"verdict":"none","reason":"Assistant contradicted the country"}]
""".strip()


def render_dialog(turns) -> str:
    return "\n".join(
        f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
        for i, t in enumerate(turns)
    )


def extract_json_block(text: str) -> str:
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def softmax_agg(scores: List[float], temperature: float = 1.1) -> float:
    if not scores:
        return 0.0
    exp_scores = [exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    return sum(s * e / total for s, e in zip(scores, exp_scores))


class KnowledgeRetentionMetric(ConversationalMetricPattern):

    name = "knowledgeRetentionMetric"
    template_cls = None  # не нужен отдельный шаблонный класс

    async def _generate_verdicts(self, dialogue: str) -> Tuple[List[Dict[str, str]], float, float]:
        prompt = prompt = (
            f"{LABEL_HELP}\n\n"
            f"{FEW_SHOT}\n\n"
            "Now analyse the next conversation.\n\n"
            f"Conversation:\n{dialogue}\n\n"
            "Return ONE JSON array with 1 object in the form:\n"
            "[{\"verdict\":\"fully|mostly|partial|minor|none\",\"reason\":\"…\"}]"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        verdicts = json.loads(extract_json_block(text))

        weights = [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in verdicts]
        score = round(softmax_agg(weights), 4)
        return verdicts, score, cost or 0.0

    async def _summarize(self, verdicts: List[Dict[str, str]]) -> Tuple[str, float]:
        bullets = "\n".join(f"- {v['reason']}" for v in verdicts)
        prompt = (
            "Write a concise (max 2 sentences) summary of the assistant’s knowledge retention, "
            "based on these points:\n\n"
            f"{bullets}\n\nSummary:"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return text.strip(), cost or 0.0

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        dialogue_text = render_dialog(test_case.turns)

        # Verdicts
        verdicts, score, c = await self._generate_verdicts(dialogue_text)
        llm_cost += c

        # Summary
        summary, c = await self._summarize(verdicts)
        llm_cost += c

        success = score >= self.threshold

        evaluation_log = {
            "dialogue": dialogue_text,
            "comment_dialogue": "Dialogue used to evaluate knowledge retention.",
            "verdicts": verdicts,
            "comment_verdicts": "LLM verdicts based on retention of prior knowledge.",
            "final_score": score,
            "comment_final_score": "Softmax aggregation of verdicts.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "True if final_score ≥ threshold.",
            "final_reason": summary,
            "comment_reasoning": "Concise explanation of how well the assistant retained information.",
        }

        return {
            "score": score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log,
        }
