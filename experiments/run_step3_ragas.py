"""
Step 3: Run RAGAS evaluation on prepared datasets.
Reads from experiments/data/<name>_prepared.json
Saves to experiments/results/<name>_ragas_scores.json

Supports resume: skips already-evaluated samples and saves after each one.

Usage:
    python experiments/run_step3_ragas.py --dataset ragtruth
    python experiments/run_step3_ragas.py --all
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import dataset_path, scores_path
from experiments.eval_ragas import evaluate_single_ragas

DATASETS = ["summeval", "summeval_rel", "usr"]


def _load_existing(out_path: Path) -> dict:
    if not out_path.exists():
        return {}
    with open(out_path, "r") as f:
        data = json.load(f)
    return {row["id"]: row for row in data}


def _save(out_path: Path, results_by_id: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, indent=2, ensure_ascii=False)


def run_ragas(name: str):
    """Run RAGAS on one dataset with incremental saving."""
    data_file = dataset_path(name)
    if not data_file.exists():
        print(f"[{name}] Dataset not found at {data_file}. Run step 1 first.")
        return

    with open(data_file, "r") as f:
        samples = json.load(f)

    out = scores_path(name, "ragas")
    existing = _load_existing(out)
    skipped = 0
    errors = 0

    print(f"\n[RAGAS] Evaluating {name}: {len(samples)} samples")
    if existing:
        print(f"  Resuming: {len(existing)} already done, skipping them")

    t0 = time.time()

    for idx, sample in enumerate(samples):
        sid = sample["id"]

        if sid in existing and existing[sid].get("ragas_score", -1) >= 0:
            skipped += 1
            continue

        print(f"  [{idx+1}/{len(samples)}] {sid[:50]}...", end=" ", flush=True)

        ctx = sample.get("context", [])
        if not ctx:
            ctx = ["No context provided."]

        score = evaluate_single_ragas(
            question=sample["question"],
            answer=sample["answer"],
            context=ctx,
        )

        if score < 0:
            errors += 1

        existing[sid] = {
            "id": sid,
            "human_score": sample["human_score"],
            "ragas_score": score,
        }
        print(f"score={score:.3f}" if score >= 0 else "FAILED")

        # Save after every sample
        _save(out, existing)

    elapsed = time.time() - t0
    evaluated = len(samples) - skipped
    print(f"\n  Done in {elapsed:.1f}s. Evaluated: {evaluated}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Total results: {len(existing)}")
    print(f"  Saved -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Run RAGAS evaluation")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    names = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not names:
        parser.print_help()
        sys.exit(1)

    for name in names:
        run_ragas(name)


if __name__ == "__main__":
    main()
