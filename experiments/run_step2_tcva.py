"""
Step 2: Run TCVA evaluation on prepared datasets.
Reads from experiments/data/<name>_prepared.json
Saves to experiments/results/<name>_tcva_scores.json

Supports resume: skips already-evaluated samples and saves after each one.

Usage:
    python experiments/run_step2_tcva.py --dataset ragtruth
    python experiments/run_step2_tcva.py --all
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import dataset_path, scores_path, TEMPERATURES
from experiments.eval_tcva import evaluate_single_tcva

DATASETS = ["summeval", "summeval_rel", "usr"]


def _load_existing(out_path: Path) -> dict:
    """Load already-computed results keyed by sample id."""
    if not out_path.exists():
        return {}
    with open(out_path, "r") as f:
        data = json.load(f)
    return {row["id"]: row for row in data}


def _save(out_path: Path, results_by_id: dict):
    """Persist current results to disk."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, indent=2, ensure_ascii=False)


async def run_tcva(name: str):
    """Run TCVA on one dataset with incremental saving."""
    data_file = dataset_path(name)
    if not data_file.exists():
        print(f"[{name}] Dataset not found at {data_file}. Run step 1 first.")
        return

    with open(data_file, "r") as f:
        samples = json.load(f)

    out = scores_path(name, "tcva")
    existing = _load_existing(out)
    skipped = 0
    errors = 0

    print(f"\n[TCVA] Evaluating {name}: {len(samples)} samples")
    print(f"  Temperatures: {TEMPERATURES}")
    if existing:
        print(f"  Resuming: {len(existing)} already done, skipping them")

    t0 = time.time()

    for idx, sample in enumerate(samples):
        sid = sample["id"]

        # Skip already evaluated
        if sid in existing and "error" not in existing[sid]:
            skipped += 1
            continue

        print(f"  [{idx+1}/{len(samples)}] {sid[:50]}...", end=" ", flush=True)

        try:
            res = await evaluate_single_tcva(
                question=sample["question"],
                answer=sample["answer"],
                context=sample.get("context", []),
                metric_type=sample.get("metric_type", "faithfulness"),
            )
            row = {
                "id": sid,
                "human_score": sample["human_score"],
                "tcva_scores": res["scores"],
                "ablation_arithmetic": res["ablation_arithmetic"],
                "verdicts": res["verdicts"],
                "weights": res["weights"],
            }
            existing[sid] = row
            s = res["scores"]
            print(f"T0.2={s.get('T0.2',0):.3f} T0.5={s.get('T0.5',0):.3f} "
                  f"T0.7={s.get('T0.7',0):.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            existing[sid] = {
                "id": sid,
                "human_score": sample["human_score"],
                "tcva_scores": {f"T{T}": -1.0 for T in TEMPERATURES},
                "ablation_arithmetic": -1.0,
                "verdicts": [],
                "weights": [],
                "error": str(e),
            }

        # Save after every sample
        _save(out, existing)

    elapsed = time.time() - t0
    evaluated = len(samples) - skipped
    print(f"\n  Done in {elapsed:.1f}s. Evaluated: {evaluated}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Total results: {len(existing)}")
    print(f"  Saved -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Run TCVA evaluation")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    names = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not names:
        parser.print_help()
        sys.exit(1)

    for name in names:
        asyncio.run(run_tcva(name))


if __name__ == "__main__":
    main()
