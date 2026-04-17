"""
Perturbation Generator - Based on Meshkov (2026) metamorphic testing

Generates variations of input queries for robustness testing.

Supports four types of perturbations (from Meshkov's taxonomy):
1. Semantic paraphrases - preserve meaning, change wording
2. Syntactic variations - change sentence structure
3. Noise injection - typos, extra spaces, case changes
4. Contextual additions - add polite forms, clarifications

Can use LLM for semantic/syntactic perturbations or apply
deterministic transformations for noise injection.

Usage:
    from eval_lib.reliability_metrics.perturbation_generator import PerturbationGenerator

    gen = PerturbationGenerator()

    # Deterministic (no LLM needed)
    noisy = gen.add_noise("Find all meetings next week")
    # → "find  all metings next week"

    # LLM-based (requires model)
    paraphrases = await gen.generate_paraphrases(
        "Find all meetings next week", n=3, model="gpt-4o-mini"
    )
"""

import random
import string
import uuid
from typing import List, Optional
from eval_lib.testcases_schema import EvalTestCase


class PerturbationGenerator:
    """Generates perturbed versions of inputs for robustness testing."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def generate_noise_variants(
        self,
        test_case: EvalTestCase,
        n: int = 3,
        group_id: Optional[str] = None,
    ) -> List[EvalTestCase]:
        """Generate n noise-injected variants of a test case.

        Deterministic — no LLM needed.
        """
        group = group_id or str(uuid.uuid4())
        variants = []

        # Original with group tag
        original = test_case.model_copy()
        original.perturbation_group = group
        variants.append(original)

        for i in range(n):
            noisy_input = self.add_noise(test_case.input)
            variant = test_case.model_copy()
            variant.input = noisy_input
            variant.perturbation_group = group
            variant.name = f"{test_case.name or 'test'}_noise_{i}" if test_case.name else f"noise_{i}"
            variants.append(variant)

        return variants

    def add_noise(self, text: str) -> str:
        """Apply random noise to text: typos, case changes, extra spaces."""
        transforms = [
            self._add_typos,
            self._change_case,
            self._add_extra_spaces,
            self._remove_punctuation,
        ]
        # Apply 1-2 random transforms
        n_transforms = self._rng.randint(1, 2)
        selected = self._rng.sample(transforms, min(n_transforms, len(transforms)))
        result = text
        for transform in selected:
            result = transform(result)
        return result

    def generate_contextual_variants(
        self,
        test_case: EvalTestCase,
        group_id: Optional[str] = None,
    ) -> List[EvalTestCase]:
        """Generate contextual addition variants. Deterministic."""
        group = group_id or str(uuid.uuid4())
        prefixes = [
            "Could you please ",
            "I would like you to ",
            "Please help me ",
            "I need you to ",
        ]
        suffixes = [
            "",
            " Thanks!",
            " Please be thorough.",
            " This is urgent.",
        ]

        variants = []
        original = test_case.model_copy()
        original.perturbation_group = group
        variants.append(original)

        for i, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
            inp = test_case.input
            # Add prefix (lowercase first char of original if adding prefix)
            if prefix and inp and inp[0].isupper():
                new_input = prefix + inp[0].lower() + inp[1:] + suffix
            else:
                new_input = prefix + inp + suffix

            variant = test_case.model_copy()
            variant.input = new_input.strip()
            variant.perturbation_group = group
            variant.name = f"{test_case.name or 'test'}_ctx_{i}" if test_case.name else f"ctx_{i}"
            variants.append(variant)

        return variants

    def _add_typos(self, text: str) -> str:
        """Introduce 1-2 character-level typos."""
        if len(text) < 5:
            return text
        chars = list(text)
        n_typos = self._rng.randint(1, min(2, len(chars) // 5))
        for _ in range(n_typos):
            pos = self._rng.randint(1, len(chars) - 2)
            if chars[pos].isalpha():
                action = self._rng.choice(["swap", "delete", "insert"])
                if action == "swap" and pos + 1 < len(chars):
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                elif action == "delete":
                    chars.pop(pos)
                elif action == "insert":
                    chars.insert(pos, self._rng.choice(string.ascii_lowercase))
        return "".join(chars)

    def _change_case(self, text: str) -> str:
        """Randomly change case of the text."""
        choice = self._rng.choice(["lower", "first_lower"])
        if choice == "lower":
            return text.lower()
        else:
            return text[0].lower() + text[1:] if text else text

    def _add_extra_spaces(self, text: str) -> str:
        """Add random extra spaces."""
        words = text.split()
        result = []
        for w in words:
            if self._rng.random() < 0.2:
                result.append("  " + w)
            else:
                result.append(w)
        return " ".join(result)

    def _remove_punctuation(self, text: str) -> str:
        """Remove some punctuation."""
        if not text:
            return text
        # Remove trailing punctuation
        if text[-1] in ".?!":
            return text[:-1]
        return text
