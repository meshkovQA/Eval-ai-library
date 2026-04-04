"""
Step 1: Download and prepare datasets.
Saves normalized samples to experiments/data/<name>_prepared.json

All datasets use stratified sampling to ensure balanced score distribution.

Usage:
    python experiments/run_step1_load.py --dataset summeval --limit 200
    python experiments/run_step1_load.py --dataset openai_sff --limit 200
    python experiments/run_step1_load.py --dataset usr --limit 200
    python experiments/run_step1_load.py --all --limit 200
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config import dataset_path
from experiments.dataset_loaders import (
    load_summeval, load_summeval_relevance, load_usr,
)

LOADERS = {
    "summeval": load_summeval,
    "summeval_rel": load_summeval_relevance,
    "usr": load_usr,
}


def main():
    parser = argparse.ArgumentParser(description="Step 1: Load and prepare datasets")
    parser.add_argument("--dataset", choices=list(LOADERS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    names = list(LOADERS.keys()) if args.all else ([args.dataset] if args.dataset else [])
    if not names:
        parser.print_help()
        sys.exit(1)

    for name in names:
        samples = LOADERS[name](limit=args.limit)
        if not samples:
            print(f"  [{name}] No samples loaded, skipping.\n")
            continue

        out = dataset_path(name)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(samples)} samples -> {out}")

        # Print distribution summary
        domains = {}
        for s in samples:
            domains[s["domain"]] = domains.get(s["domain"], 0) + 1
        scores = [s["human_score"] for s in samples]
        print(f"  Domains: {domains}")
        print(f"  Human score range: [{min(scores):.3f}, {max(scores):.3f}], "
              f"mean={sum(scores)/len(scores):.3f}")
        print()


if __name__ == "__main__":
    main()
