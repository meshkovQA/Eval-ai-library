"""
Step 7: A/B test old vs new prompts on 20 worst-case examples.

Compares the current (old) prompts with improved prompts on examples
where TCVA gave incorrect scores (human=1.0 but TCVA<<1.0 due to false 'none' verdicts).

Usage:
    python experiments/run_step7_ab_test_prompts.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block
from experiments.config import EVAL_MODEL, VERDICT_WEIGHTS

# IDs of worst cases (human>=0.95 but TCVA low due to false 'none')
TEST_IDS = ['2442', '1107', '46', '1455', '68', '616', '180', '317',
            '467', '2168', '332', '605', '1730', '1053', '1521', '844',
            '2137', '1480', '1682', '1732']


# ---- OLD prompts (current) ----

async def old_extract_statements(answer: str) -> List[str]:
    prompt = (
        "Extract standalone factual claims from the following answer.\n"
        "Each statement must be a distinct, verifiable fact.\n\n"
        f"Answer:\n{answer}\n\n"
        "Return a JSON array of strings."
    )
    text, _ = await chat_complete(EVAL_MODEL, [{"role": "user", "content": prompt}], temperature=0.0)
    return json.loads(extract_json_block(text))


async def old_generate_verdicts(context: str, statements: List[str]) -> List[Dict]:
    prompt = (
        "Evaluate how well each statement is supported by the context.\n\n"
        "Levels:\n"
        "- fully: directly supported word-for-word\n"
        "- mostly: strongly supported but wording differs slightly\n"
        "- partial: partially supported but with some gaps\n"
        "- minor: tangentially related or ambiguous\n"
        "- none: clearly unsupported or contradicted\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"STATEMENTS (JSON array):\n{json.dumps(statements, ensure_ascii=False)}\n\n"
        "Return only a JSON array of objects like:\n"
        '[{"verdict": "fully|mostly|partial|minor|none", '
        '"reason": "<brief>", '
        '"support": "<exact context sentence(s)> or \'none\'"}]'
    )
    text, _ = await chat_complete(EVAL_MODEL, [{"role": "user", "content": prompt}], temperature=0.0)
    return json.loads(extract_json_block(text))


# ---- NEW prompts (improved: two-step verdict) ----

async def new_extract_statements(answer: str) -> List[str]:
    prompt = (
        "Extract the key factual claims from the following answer.\n\n"
        "Rules:\n"
        "- Each claim must be a single, verifiable factual statement.\n"
        "- Ignore greetings, meta-comments (\"Sure!\", \"Here's...\"), and stylistic phrases.\n"
        "- Do NOT split one sentence into micro-facts. Keep claims at sentence-level granularity.\n"
        "- Combine closely related details into one claim rather than listing separately.\n"
        "- Maximum 8 claims. Focus on the most important facts.\n\n"
        f"Answer:\n{answer}\n\n"
        "Return a JSON array of strings."
    )
    text, _ = await chat_complete(EVAL_MODEL, [{"role": "user", "content": prompt}], temperature=0.0)
    return json.loads(extract_json_block(text))


async def new_generate_verdicts(context: str, statements: List[str]) -> List[Dict]:
    """Two-step verdict: coarse classification → fine-grained level."""

    # Step 1: Coarse (yes / partially / no)
    step1_prompt = (
        "For each statement, determine whether it is supported by the context.\n\n"
        "Answer with ONE of:\n"
        "- yes: The statement's meaning is present in the context (even if paraphrased).\n"
        "- partially: Some parts are supported, but other parts are missing or vague.\n"
        "- no: The statement contradicts the context, OR the context has no related information at all.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"STATEMENTS:\n{json.dumps(statements, ensure_ascii=False)}\n\n"
        "Return a JSON array of objects:\n"
        '[{"classification": "yes|partially|no", '
        '"support": "<relevant context quote or \'none\'>"}]'
    )
    text, _ = await chat_complete(EVAL_MODEL, [{"role": "user", "content": step1_prompt}], temperature=0.0)
    step1_raw = json.loads(extract_json_block(text))

    # Step 2: Fine-grained
    step2_prompt = (
        "You previously classified each statement as yes/partially/no.\n"
        "Now assign a precise verdict level for each.\n\n"
        "For statements classified as 'yes':\n"
        "- fully: The core meaning matches the context (paraphrasing is OK).\n"
        "- mostly: The main idea is correct but minor details differ.\n\n"
        "For statements classified as 'partially':\n"
        "- partial: Some parts are supported but key information is missing.\n"
        "- minor: Only the general topic is mentioned; the specific claim is not addressed.\n\n"
        "For statements classified as 'no':\n"
        "- none_contradicts: The context explicitly says something different.\n"
        "- none_absent: The context simply does not mention this topic at all.\n\n"
        f"STATEMENTS:\n{json.dumps(statements, ensure_ascii=False)}\n\n"
        f"STEP 1 CLASSIFICATIONS:\n{json.dumps(step1_raw, ensure_ascii=False)}\n\n"
        "Return a JSON array of objects:\n"
        '[{"verdict": "<level>", "reason": "<brief>"}]'
    )
    text, _ = await chat_complete(EVAL_MODEL, [{"role": "user", "content": step2_prompt}], temperature=0.0)
    step2_raw = json.loads(extract_json_block(text))

    # Merge
    verdicts = []
    for i in range(len(statements)):
        s1 = step1_raw[i] if i < len(step1_raw) else {}
        s2 = step2_raw[i] if i < len(step2_raw) else {}

        raw_verdict = s2.get("verdict", "none_absent").strip().lower()
        reason = s2.get("reason", "")
        support = s1.get("support", "none")

        if raw_verdict in ("fully", "mostly", "partial", "minor", "none"):
            verdict = raw_verdict
        elif raw_verdict == "none_contradicts":
            verdict = "none"
        elif raw_verdict == "none_absent":
            verdict = "minor"  # absent ≠ contradicts
        else:
            classification = s1.get("classification", "no").strip().lower()
            if classification == "yes":
                verdict = "mostly"
            elif classification == "partially":
                verdict = "partial"
            else:
                verdict = "minor"

        # Safety check
        supp = str(support).strip().lower()
        if supp in ("none", "") and verdict in ("fully", "mostly"):
            verdict = "partial"

        verdicts.append({
            "verdict": verdict,
            "reason": str(reason)[:200],
            "support": str(support)[:300],
        })

    return verdicts


async def run_pipeline(extract_fn, verdict_fn, question, answer, context) -> Dict:
    """Run one full pipeline and return results."""
    try:
        statements = await extract_fn(answer)
        verdicts = await verdict_fn(context, statements)

        # Apply support-check fix
        for v in verdicts:
            supp = v.get("support", "").strip().lower()
            if supp == "none" and v["verdict"] in ("fully", "mostly"):
                v["verdict"] = "partial"

        weights = [VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in verdicts]
        score = score_agg(weights, temperature=0.5) if weights else 0.0

        return {
            "n_statements": len(statements),
            "verdicts": [(v["verdict"], v.get("reason", "")[:60]) for v in verdicts],
            "weights": weights,
            "score_T0.5": round(score, 4),
            "none_count": sum(1 for v in verdicts if v["verdict"] == "none"),
            "minor_count": sum(1 for v in verdicts if v["verdict"] == "minor"),
        }
    except Exception as e:
        return {"error": str(e), "score_T0.5": -1.0}


async def main():
    # Load dataset
    dataset = json.load(open("experiments/data/ragtruth_prepared.json"))
    samples = {str(d["id"]): d for d in dataset}

    # Load old results for comparison
    old_results = json.load(open("experiments/results/ragtruth_tcva_scores.json"))
    old_by_id = {str(r["id"]): r for r in old_results}

    print(f"A/B Test: Old vs New prompts on {len(TEST_IDS)} worst cases")
    print(f"Model: {EVAL_MODEL}")
    print(f"{'='*90}")

    improvements = []

    for sid in TEST_IDS:
        sample = samples.get(sid)
        if not sample:
            print(f"  [SKIP] ID={sid} not in dataset")
            continue

        old_data = old_by_id.get(sid, {})
        old_score = old_data.get("tcva_scores", {}).get("T0.5", -1)
        human = sample["human_score"]

        print(f"\nID={sid} | human={human:.2f} | old_score={old_score:.3f}")

        # Run new pipeline
        context = "\n".join(sample.get("context", []))
        new_result = await run_pipeline(
            new_extract_statements, new_generate_verdicts,
            sample["question"], sample["answer"], context,
        )

        if "error" in new_result:
            print(f"  ERROR: {new_result['error']}")
            continue

        new_score = new_result["score_T0.5"]
        old_err = abs(human - old_score)
        new_err = abs(human - new_score)
        diff = old_err - new_err  # positive = new is better

        marker = "BETTER" if diff > 0.05 else ("WORSE" if diff < -0.05 else "SAME")

        print(f"  Old: score={old_score:.3f} err={old_err:.3f} | "
              f"stmts={old_data.get('verdicts', []) and len(old_data.get('verdicts', []))} "
              f"nones={sum(1 for v in old_data.get('verdicts',[]) if v.get('verdict')=='none')}")
        print(f"  New: score={new_score:.3f} err={new_err:.3f} | "
              f"stmts={new_result['n_statements']} "
              f"nones={new_result['none_count']} → [{marker}]")

        if new_result["none_count"] > 0:
            for verd, reason in new_result["verdicts"]:
                if verd in ("none", "minor"):
                    print(f"    {verd}: {reason}")

        improvements.append({
            "id": sid, "human": human,
            "old_score": old_score, "new_score": new_score,
            "old_err": old_err, "new_err": new_err,
            "improvement": diff, "marker": marker,
        })

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    better = sum(1 for i in improvements if i["marker"] == "BETTER")
    worse = sum(1 for i in improvements if i["marker"] == "WORSE")
    same = sum(1 for i in improvements if i["marker"] == "SAME")
    avg_old_err = sum(i["old_err"] for i in improvements) / len(improvements)
    avg_new_err = sum(i["new_err"] for i in improvements) / len(improvements)

    print(f"  Better: {better}/{len(improvements)}")
    print(f"  Worse:  {worse}/{len(improvements)}")
    print(f"  Same:   {same}/{len(improvements)}")
    print(f"  Avg old error: {avg_old_err:.4f}")
    print(f"  Avg new error: {avg_new_err:.4f}")
    print(f"  Improvement:   {avg_old_err - avg_new_err:+.4f}")

    # Save
    out = Path("experiments/results/ab_test_prompts.json")
    with open(out, "w") as f:
        json.dump(improvements, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    asyncio.run(main())
