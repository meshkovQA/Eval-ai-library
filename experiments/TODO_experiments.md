# Experiment Plan for Article Revision

Priority order for strengthening the paper. Items 1-2 require no new LLM calls (recomputation only). Items 3-4 require new LLM calls and budget.

## Priority 1: Sensitivity Analysis for Weights [NO LLM CALLS]
**Goal**: Show results are robust to weight selection {1.0, 0.9, 0.7, 0.3, 0.0}.
**Method**: Recompute score_agg() with 4 weight schemes on stored verdicts across 3 datasets.
**Weight schemes**:
- Default: {1.0, 0.9, 0.7, 0.3, 0.0}
- Linear: {1.0, 0.75, 0.5, 0.25, 0.0}
- Aggressive: {1.0, 0.95, 0.8, 0.1, 0.0}
- Conservative: {1.0, 0.8, 0.5, 0.2, 0.0}
**Output**: Table of Spearman's rho per weight scheme x dataset. Target: variation < 0.03.
**Status**: NOT STARTED

## Priority 2: Bootstrap Confidence Intervals [NO LLM CALLS]
**Goal**: Provide 95% CI for all Spearman's rho values; test statistical significance of differences.
**Method**: 10,000 bootstrap resamples on existing scores.
**Output**: CI for each method x dataset. Paired bootstrap test TCVA vs RAGAS.
**Status**: NOT STARTED

## Priority 3: Ablation Study [REQUIRES LLM CALLS]
**Goal**: Show contribution of each TCVA component independently.
**Configurations**:
- A. Full TCVA (5-level + power mean + penalty)
- B. TCVA w/o penalty (5-level + power mean, no Step 6)
- C. 5-level + arithmetic mean (fix T=0.5, no power mean variation)
- D. Binary verdicts + power mean (new prompts needed for binary verdicts)
**Note**: Config B and C can be computed from existing data. Config D requires new LLM calls with binary prompts.
**Status**: NOT STARTED

## Priority 4: Multi-Judge Experiment [REQUIRES LLM CALLS]
**Goal**: Show TCVA is robust across judge models.
**Models**: GPT-4o, Claude Sonnet 4, Llama-3.1-70B (via Groq)
**Note**: Full re-evaluation of all 3 datasets with each model. ~3x current token cost.
**Status**: NOT STARTED
