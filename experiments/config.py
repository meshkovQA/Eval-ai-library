"""
Shared configuration for the TCVA experiment.
"""
import os
from pathlib import Path

# LLM model used as judge in all methods
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4.1-mini")

# TCVA temperatures to test
TEMPERATURES = [0.2, 0.3, 0.5, 0.7, 0.9]

# Verdict weight mapping (same as eval_lib)
VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def dataset_path(name: str) -> Path:
    """Path to the prepared dataset JSON."""
    return DATA_DIR / f"{name}_prepared.json"


def scores_path(name: str, method: str) -> Path:
    """Path to scores produced by a specific method."""
    return RESULTS_DIR / f"{name}_{method}_scores.json"
