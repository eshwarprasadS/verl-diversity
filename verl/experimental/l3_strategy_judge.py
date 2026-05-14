"""L3 strategy diversity judge using OpenAI API.

Classifies reasoning strategies across G rollouts for a group of math solutions.
Called from _compute_diversity_metrics during training.

Uses coarse strategy categories to ensure reliable classification:
- algebraic: manipulation, substitution, equation solving
- arithmetic: direct computation, enumeration, brute force
- geometric: visual/spatial reasoning, coordinate geometry
- analytic: calculus, limits, series, optimization
- combinatorial: counting, probability, permutations
- number_theoretic: modular arithmetic, divisibility, primes
- logical: proof by contradiction, cases, induction
- other: anything not fitting above

Design choices:
- Coarse categories (8) rather than fine-grained to maximize inter-rater reliability
- All G rollouts sent in one call for comparative judgment
- JSON mode for structured output
- 10s timeout to not block training
- Failures are silent (return None, training continues)
"""

import json
import os
from typing import Optional

L3_JUDGE_MODEL = "gpt-5.4-mini"
L3_SAMPLE_GROUPS = 16
L3_TIMEOUT = 10

L3_SYSTEM_PROMPT = """You are a math reasoning strategy classifier. Given multiple solution attempts to the same math problem, classify each solution's primary reasoning strategy.

Use ONLY these coarse categories:
- algebraic: equation manipulation, substitution, factoring, polynomial operations
- arithmetic: direct computation, enumeration, brute force calculation
- geometric: spatial/visual reasoning, coordinate geometry, diagrams
- analytic: calculus, limits, series, continuous optimization
- combinatorial: counting principles, permutations, combinations, probability
- number_theoretic: modular arithmetic, divisibility, prime factorization, GCD/LCM
- logical: proof by contradiction, case analysis, induction, pigeonhole
- other: anything not fitting the above categories

Focus on the PRIMARY approach, not minor sub-steps. Two solutions that both use algebra but with different specific manipulations are STILL "algebraic" — same strategy.

Respond with JSON only: {"strategies": ["algebraic", "algebraic", "number_theoretic", ...], "num_distinct": 2}
The strategies array must have exactly one entry per solution, in the same order as presented."""

L3_USER_TEMPLATE = """Problem: {problem}

{solutions_text}

Classify each solution's primary reasoning strategy. Return JSON with "strategies" (array of category labels, one per solution) and "num_distinct" (count of unique strategies used)."""


def classify_strategies(problem_text: str, solution_texts: list[str],
                        api_key: Optional[str] = None) -> Optional[dict]:
    try:
        import openai
    except ImportError:
        return None

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None

    solutions_text = "\n\n".join(
        f"=== Solution {i+1} ===\n{text[:2000]}"
        for i, text in enumerate(solution_texts)
    )

    try:
        client = openai.OpenAI(api_key=key, timeout=L3_TIMEOUT)
        response = client.chat.completions.create(
            model=L3_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": L3_SYSTEM_PROMPT},
                {"role": "user", "content": L3_USER_TEMPLATE.format(
                    problem=problem_text[:500],
                    solutions_text=solutions_text,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
        result = json.loads(response.choices[0].message.content)
        if "num_distinct" in result and "strategies" in result:
            return result
    except Exception:
        pass
    return None


def compute_l3_metrics(groups: list[dict], tokenizer, api_key: Optional[str] = None) -> dict:
    """Compute L3 strategy diversity for a sample of groups.

    Args:
        groups: list of dicts with 'problem' (str) and 'responses' (tensor of token ids)
        tokenizer: for decoding responses
        api_key: OpenAI API key

    Returns:
        dict with diversity/L3_strategies_per_group and diversity/L3_strategy_entropy
    """
    import numpy as np

    if not groups:
        return {}

    sampled = groups[:L3_SAMPLE_GROUPS]
    strategies_per_group = []
    strategy_entropies = []

    for group in sampled:
        solution_texts = [
            tokenizer.decode(resp, skip_special_tokens=True)
            for resp in group["responses"]
        ]
        result = classify_strategies(group["problem"], solution_texts, api_key)
        if result is None:
            continue

        n_distinct = result.get("num_distinct", 1)
        strategies_per_group.append(n_distinct)

        labels = result.get("strategies", [])
        if labels:
            from collections import Counter
            counts = Counter(labels)
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            strategy_entropies.append(entropy)

    metrics = {}
    if strategies_per_group:
        metrics["diversity/L3_strategies_per_group"] = float(np.mean(strategies_per_group))
    if strategy_entropies:
        metrics["diversity/L3_strategy_entropy"] = float(np.mean(strategy_entropies))
    return metrics
