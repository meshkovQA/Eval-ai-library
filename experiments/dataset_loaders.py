"""
Dataset loaders: download and normalize datasets into a common format.

Each sample: {
    "id": str,
    "question": str,
    "answer": str,
    "context": List[str],
    "human_score": float (0-1),
    "domain": str,
    "dataset": str,
    "metric_type": "faithfulness" | "answer_relevancy",
}
"""
import json
import random
from typing import List, Dict, Any

from datasets import load_dataset


def stratified_sample(samples: List[Dict], limit: int, n_bins: int = 5, seed: int = 42) -> List[Dict]:
    """Select a balanced stratified sample based on human_score.

    Splits samples into n_bins equal-width bins by human_score,
    then takes limit/n_bins samples from each bin (or all if fewer).
    This prevents the dataset from being dominated by one score range.
    """
    if len(samples) <= limit:
        return samples

    random.seed(seed)

    # Create bins
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins: List[List[Dict]] = [[] for _ in range(n_bins)]

    for s in samples:
        score = s["human_score"]
        # Find which bin this score belongs to
        bin_idx = min(int(score * n_bins), n_bins - 1)
        bins[bin_idx].append(s)

    # Report bin sizes
    for i, b in enumerate(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        print(f"    bin [{lo:.1f}-{hi:.1f}): {len(b)} samples")

    # Sample equally from each non-empty bin
    non_empty = [b for b in bins if b]
    per_bin = limit // len(non_empty)
    remainder = limit - per_bin * len(non_empty)

    result = []
    for i, b in enumerate(non_empty):
        take = per_bin + (1 if i < remainder else 0)
        if len(b) <= take:
            result.extend(b)
        else:
            result.extend(random.sample(b, take))

    random.shuffle(result)
    print(f"    stratified: {len(result)} samples from {len(non_empty)} bins")
    return result


def load_ragtruth(limit: int = 200) -> List[Dict[str, Any]]:
    """
    RAGTruth: span-level hallucination annotations.
    Human score = 1 - (fraction of hallucinated characters in response).
    """
    print("[RAGTruth] Loading from HuggingFace...")
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    samples = []

    for row in ds:
        if len(samples) >= limit:
            break
        if row.get("quality") not in ("good", None):
            continue

        query = (row.get("query") or "").strip()
        output = (row.get("output") or "").strip()
        context = (row.get("context") or "").strip()

        if not query or not output or not context:
            continue

        # Parse hallucination span annotations
        labels = row.get("hallucination_labels", "[]")
        if isinstance(labels, str):
            try:
                spans = json.loads(labels)
            except json.JSONDecodeError:
                spans = []
        else:
            spans = labels or []

        total_chars = max(len(output), 1)
        halluc_chars = sum(abs(s.get("end", 0) - s.get("start", 0)) for s in spans)
        human_score = round(max(0.0, 1.0 - halluc_chars / total_chars), 4)

        samples.append({
            "id": row.get("id", f"ragtruth_{len(samples)}"),
            "question": query,
            "answer": output,
            "context": [context],
            "human_score": human_score,
            "domain": row.get("task_type", "QA"),
            "dataset": "RAGTruth",
            "metric_type": "faithfulness",
        })

    print(f"[RAGTruth] Loaded {len(samples)} samples")
    return samples


def load_halueval_qa(limit: int = 200) -> List[Dict[str, Any]]:
    """
    HaluEval QA: 10K QA samples with binary hallucination labels.
    knowledge (context) + question + answer + hallucination (yes/no).
    Human score: 1.0 if no hallucination, 0.0 if hallucinated.
    """
    print("[HaluEval-QA] Loading from HuggingFace...")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    samples = []

    for row in ds:
        if len(samples) >= limit:
            break

        knowledge = (row.get("knowledge") or "").strip()
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        halluc = (row.get("hallucination") or "").strip().lower()

        if not question or not answer or not knowledge:
            continue

        human_score = 0.0 if halluc == "yes" else 1.0

        samples.append({
            "id": f"halueval_qa_{len(samples)}",
            "question": question,
            "answer": answer,
            "context": [knowledge],
            "human_score": human_score,
            "domain": "QA",
            "dataset": "HaluEval-QA",
            "metric_type": "faithfulness",
        })

    print(f"[HaluEval-QA] Loaded {len(samples)} samples")
    return samples


def load_halueval_summarization(limit: int = 200) -> List[Dict[str, Any]]:
    """
    HaluEval Summarization: 10K summarization samples with binary hallucination labels.
    document (context) + summary (answer) + hallucination (yes/no).
    """
    print("[HaluEval-Summ] Loading from HuggingFace...")
    ds = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
    samples = []

    for row in ds:
        if len(samples) >= limit:
            break

        document = (row.get("document") or "").strip()
        summary = (row.get("summary") or "").strip()
        halluc = (row.get("hallucination") or "").strip().lower()

        if not document or not summary:
            continue

        human_score = 0.0 if halluc == "yes" else 1.0

        samples.append({
            "id": f"halueval_summ_{len(samples)}",
            "question": "Summarize the following document.",
            "answer": summary,
            "context": [document[:4000]],
            "human_score": human_score,
            "domain": "Summarization",
            "dataset": "HaluEval-Summ",
            "metric_type": "faithfulness",
        })

    print(f"[HaluEval-Summ] Loaded {len(samples)} samples")
    return samples


def load_halueval_dialogue(limit: int = 200) -> List[Dict[str, Any]]:
    """
    HaluEval Dialogue: 10K dialogue samples with binary hallucination labels.
    knowledge (context) + dialogue_history (question) + response (answer) + hallucination.
    """
    print("[HaluEval-Dial] Loading from HuggingFace...")
    ds = load_dataset("pminervini/HaluEval", "dialogue_samples", split="data")
    samples = []

    for row in ds:
        if len(samples) >= limit:
            break

        knowledge = (row.get("knowledge") or "").strip()
        history = (row.get("dialogue_history") or "").strip()
        response = (row.get("response") or "").strip()
        halluc = (row.get("hallucination") or "").strip().lower()

        if not history or not response:
            continue

        human_score = 0.0 if halluc == "yes" else 1.0

        samples.append({
            "id": f"halueval_dial_{len(samples)}",
            "question": history,
            "answer": response,
            "context": [knowledge] if knowledge else [],
            "human_score": human_score,
            "domain": "Dialogue",
            "dataset": "HaluEval-Dial",
            "metric_type": "faithfulness",
        })

    print(f"[HaluEval-Dial] Loaded {len(samples)} samples")
    return samples


def load_expertqa(limit: int = 200) -> List[Dict[str, Any]]:
    """
    ExpertQA: expert-curated QA across 32 fields with 5-point correctness scores.

    Structure: Each question has multiple answer variants (gpt4, bing_chat, etc.).
    Each answer has claim-level expert annotations with 'correctness' field.

    For faithfulness testing: we use one answer variant as the "answer" and
    construct context from the claim_strings of the best-scored variant.
    This creates a natural faithfulness test — does the answer align with
    expert-verified claims?

    Score: 5-point Likert aggregated from claim-level correctness.
    """
    print("[ExpertQA] Loading from HuggingFace...")
    ds = load_dataset("cmalaviya/expertqa", "main", split="train")
    samples = []

    CORRECTNESS_MAP = {
        "definitely correct": 1.0,
        "probably correct": 0.75,
        "unsure": 0.5,
        "likely incorrect": 0.25,
        "definitely incorrect": 0.0,
    }

    for row in ds:
        if len(samples) >= limit:
            break

        question = (row.get("question") or "").strip()
        if not question:
            continue

        answers = row.get("answers", {})
        if not answers:
            continue

        # Find the answer variant with the most claims
        best_key = None
        best_claims = []
        best_answer = ""

        for ans_key, ans_data in answers.items():
            if ans_data is None:
                continue
            claims = ans_data.get("claims", [])
            answer_str = (ans_data.get("answer_string") or "").strip()
            if claims and answer_str and len(claims) > len(best_claims):
                best_key = ans_key
                best_claims = claims
                best_answer = answer_str

        if not best_claims or not best_answer:
            continue

        # Aggregate correctness scores
        claim_scores = []
        claim_texts = []
        for claim in best_claims:
            if not isinstance(claim, dict):
                continue
            correctness = (claim.get("correctness") or "").strip().lower()
            score = CORRECTNESS_MAP.get(correctness)
            if score is not None:
                claim_scores.append(score)

            # Use claim_string as context (expert-verified claims)
            claim_str = (claim.get("claim_string") or "").strip()
            if claim_str:
                claim_texts.append(claim_str)

        if not claim_scores or not claim_texts:
            continue

        human_score = round(sum(claim_scores) / len(claim_scores), 4)

        # Context = expert-verified claim strings
        # This creates a faithfulness test: does the full answer
        # accurately represent these verified claims?
        metadata = row.get("metadata", {})
        field = metadata.get("field", "general") if isinstance(metadata, dict) else "general"

        samples.append({
            "id": f"expertqa_{len(samples)}",
            "question": question,
            "answer": best_answer[:3000],
            "context": claim_texts,
            "human_score": human_score,
            "domain": field,
            "dataset": "ExpertQA",
            "metric_type": "faithfulness",
        })

    print(f"[ExpertQA] Loaded {len(samples)} samples")
    return samples


def load_frank(limit: int = 200) -> List[Dict[str, Any]]:
    """
    FRANK: fine-grained factuality annotations for summarization.

    Two files needed:
    - benchmark_data.json: article + summary + hash
    - human_annotations.json: hash + Factuality score (0-1 continuous)

    Joined by hash. Factuality is already a continuous 0-1 score
    (fraction of annotators who judged the summary as factual).
    """
    print("[FRANK] Loading from GitHub...")

    import urllib.request
    import tempfile
    import os

    cache_dir = tempfile.gettempdir()

    # Download annotations (scores)
    annot_path = os.path.join(cache_dir, "frank_annotations.json")
    if not os.path.exists(annot_path):
        print("  Downloading human_annotations.json...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations.json",
            annot_path,
        )

    # Download benchmark data (texts)
    bench_path = os.path.join(cache_dir, "frank_benchmark.json")
    if not os.path.exists(bench_path):
        print("  Downloading benchmark_data.json...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/artidoro/frank/main/data/benchmark_data.json",
            bench_path,
        )

    with open(annot_path) as f:
        annotations = json.load(f)
    with open(bench_path) as f:
        benchmark = json.load(f)

    # Index benchmark by hash for O(1) lookup
    bench_by_hash = {}
    for row in benchmark:
        h = str(row.get("hash", ""))
        if h:
            bench_by_hash[h] = row

    samples = []

    for annot in annotations:
        if len(samples) >= limit:
            break

        h = str(annot.get("hash", ""))
        bench = bench_by_hash.get(h)
        if not bench:
            continue

        article = (bench.get("article") or "").strip()
        summary = (bench.get("summary") or "").strip()

        if not article or not summary:
            continue

        # Factuality is already 0-1 (fraction of annotators marking as factual)
        factuality = annot.get("Factuality")
        if factuality is None:
            continue

        human_score = round(float(factuality), 4)

        samples.append({
            "id": h,
            "question": "Summarize the following article.",
            "answer": summary,
            "context": [article[:4000]],
            "human_score": human_score,
            "domain": bench.get("dataset", "summarization"),
            "dataset": "FRANK",
            "metric_type": "faithfulness",
        })

    print(f"[FRANK] Loaded {len(samples)} samples")
    return samples


def load_summeval_relevance(limit: int = 200) -> List[Dict[str, Any]]:
    """
    SummEval Relevance: 'relevance' dimension (Likert 1-5).
    Loads ALL, then stratified sample to `limit`.
    """
    print("[SummEval-Relevance] Loading from HuggingFace...")
    ds = load_dataset("mteb/summeval", split="test")
    all_samples = []

    for row in ds:
        text = row.get("text", "")
        summaries = row.get("machine_summaries", [])
        relevances = row.get("relevance", [])

        if not text or not summaries:
            continue

        for i, summary in enumerate(summaries):
            if not summary.strip():
                continue
            rel_score = (relevances[i] - 1.0) / 4.0 if i < len(relevances) else 0.5
            all_samples.append({
                "id": f"summeval_rel_{row.get('id', '')}_{i}",
                "question": "Summarize the following article.",
                "answer": summary,
                "context": [text[:3000]],
                "human_score": round(rel_score, 4),
                "domain": "news_summarization",
                "dataset": "SummEval-Relevance",
                "metric_type": "answer_relevancy",
            })

    print(f"  Total loaded: {len(all_samples)}")
    samples = stratified_sample(all_samples, limit)
    print(f"[SummEval-Relevance] Final: {len(samples)} samples")
    return samples


def load_summeval(limit: int = 200) -> List[Dict[str, Any]]:
    """
    SummEval: 'consistency' dimension (Likert 1-5, human expert ratings).
    Loads ALL, then stratified sample to `limit`.
    """
    print("[SummEval] Loading from HuggingFace...")
    ds = load_dataset("mteb/summeval", split="test")
    all_samples = []

    for row in ds:
        text = row.get("text", "")
        summaries = row.get("machine_summaries", [])
        consistencies = row.get("consistency", [])

        if not text or not summaries:
            continue

        for i, summary in enumerate(summaries):
            if not summary.strip():
                continue
            cons_score = (consistencies[i] - 1.0) / 4.0 if i < len(consistencies) else 0.5
            all_samples.append({
                "id": f"summeval_{row.get('id', '')}_{i}",
                "question": "Summarize the following article.",
                "answer": summary,
                "context": [text[:3000]],
                "human_score": round(cons_score, 4),
                "domain": "news_summarization",
                "dataset": "SummEval",
                "metric_type": "faithfulness",
            })

    print(f"  Total loaded: {len(all_samples)}")
    samples = stratified_sample(all_samples, limit)
    print(f"[SummEval] Final: {len(samples)} samples")
    return samples


def load_summeval_coherence(limit: int = 200) -> List[Dict[str, Any]]:
    """
    SummEval Coherence: 'coherence' dimension (Likert 1-5).
    Tests TCVA on a third quality dimension from the same dataset.
    Loads ALL, then stratified sample.
    """
    print("[SummEval-Coherence] Loading from HuggingFace...")
    ds = load_dataset("mteb/summeval", split="test")
    all_samples = []

    for row in ds:
        text = row.get("text", "")
        summaries = row.get("machine_summaries", [])
        coherences = row.get("coherence", [])

        if not text or not summaries:
            continue

        for i, summary in enumerate(summaries):
            if not summary.strip():
                continue
            coh_score = (coherences[i] - 1.0) / 4.0 if i < len(coherences) else 0.5
            all_samples.append({
                "id": f"summeval_coh_{row.get('id', '')}_{i}",
                "question": "Summarize the following article.",
                "answer": summary,
                "context": [text[:3000]],
                "human_score": round(coh_score, 4),
                "domain": "news_summarization",
                "dataset": "SummEval-Coherence",
                "metric_type": "faithfulness",
            })

    print(f"  Total loaded: {len(all_samples)}")
    samples = stratified_sample(all_samples, limit)
    print(f"[SummEval-Coherence] Final: {len(samples)} samples")
    return samples


def load_usr(limit: int = 200) -> List[Dict[str, Any]]:
    """
    USR (Mehri & Eskenazi 2020) — dialogue evaluation.
    Graded human ratings for 'groundedness' (faithfulness in dialogue).
    Downloads from original source.
    """
    print("[USR] Loading from source...")
    import urllib.request
    import tempfile
    import os

    cache_dir = tempfile.gettempdir()

    # USR-TopicalChat data
    tc_url = "http://shikib.com/tc_usr_data.json"
    tc_path = os.path.join(cache_dir, "usr_topicalchat.json")
    if not os.path.exists(tc_path):
        print("  Downloading USR-TopicalChat...")
        urllib.request.urlretrieve(tc_url, tc_path)

    # USR-PersonaChat data
    pc_url = "http://shikib.com/pc_usr_data.json"
    pc_path = os.path.join(cache_dir, "usr_personachat.json")
    if not os.path.exists(pc_path):
        print("  Downloading USR-PersonaChat...")
        urllib.request.urlretrieve(pc_url, pc_path)

    all_samples = []

    for filepath, source in [(tc_path, "topicalchat"), (pc_path, "personachat")]:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception:
            continue

        for item in data:
            context = item.get("context", "")
            fact = item.get("fact", "") or item.get("knowledge", "")
            responses = item.get("responses", [])

            if not context:
                continue

            for resp in responses:
                response_text = resp.get("response", "")
                if not response_text:
                    continue

                # Annotations are lists of 3 annotator scores
                # "Maintains Context" = faithfulness to dialogue context (0-3 scale)
                # "Uses Knowledge" = faithfulness to knowledge/fact (0-1 binary per annotator)
                # "Overall" = overall quality (0-5 scale)
                maintains = resp.get("Maintains Context", [])
                uses_know = resp.get("Uses Knowledge", [])
                overall = resp.get("Overall", [])

                # Average annotator scores for "Maintains Context" (primary faithfulness)
                if maintains and isinstance(maintains, list):
                    avg_maintains = sum(maintains) / len(maintains)
                    # Normalize 0-3 → 0-1
                    human_score = round(avg_maintains / 3.0, 4)
                elif overall and isinstance(overall, list):
                    avg_overall = sum(overall) / len(overall)
                    # Normalize 0-5 → 0-1
                    human_score = round(avg_overall / 5.0, 4)
                else:
                    continue

                knowledge_context = fact if fact else context
                all_samples.append({
                    "id": f"usr_{source}_{len(all_samples)}",
                    "question": context[-500:],
                    "answer": response_text,
                    "context": [knowledge_context[:3000]],
                    "human_score": human_score,
                    "domain": f"dialogue_{source}",
                    "dataset": "USR",
                    "metric_type": "faithfulness",
                })

    print(f"  Total loaded: {len(all_samples)}")
    if all_samples:
        samples = stratified_sample(all_samples, limit)
    else:
        samples = []
    print(f"[USR] Final: {len(samples)} samples")
    return samples
