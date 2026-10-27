"""Turning scores into allow, review, or block

Below t_low a package is allowed without a human looking at it, at or above t_high it is
blocked, and everything in between goes to an analyst
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkgtriage.business.costs import Assumptions

ALLOW, REVIEW, BLOCK = "allow", "review", "block"
TIERS=(ALLOW, REVIEW, BLOCK)


@dataclass(frozen=True)
class TierRates:
    """Where each class ends up. Measured on labelled data, then applied to any prevalence."""

    malicious: dict[str, float]  # share of malicious packages landing in each tier
    benign: dict[str, float]

    @classmethod
    def from_tiers(cls, tiers: np.ndarray, y) -> TierRates:
        y = np.asarray(y)
        mal, ben = tiers[y == 1], tiers[y == 0]
        return cls(malicious={t: float(np.mean(mal == t)) for t in TIERS},
            benign={t: float(np.mean(ben == t)) for t in TIERS})

    @classmethod
    def everyone(cls, tier: str) -> TierRates:
        one = {t: float(t == tier) for t in TIERS}
        return cls(malicious=dict(one), benign=dict(one))


@dataclass(frozen=True)
class ThresholdPolicy:
    t_low: float
    t_high: float

    def __post_init__(self):
        if not 0 <= self.t_low <= self.t_high <= 1:
            raise ValueError(f"need 0 <= t_low <= t_high <= 1, got {self.t_low}, {self.t_high}")

    def decide(self, probabilities) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        return np.where(p >= self.t_high, BLOCK, np.where(p < self.t_low, ALLOW, REVIEW))


def expected_outcome(rates: TierRates, prevalence: float, a: Assumptions) -> dict[str, float]:
    """Expected daily numbers for a request stream with the given share of malicious packages."""
    n = a.requests_per_day
    n_mal, n_ben = n * prevalence, n * (1 - prevalence)

    reviews = n_mal * rates.malicious[REVIEW] + n_ben * rates.benign[REVIEW]
    # reviews beyond capacity don't happen today; they pile up. Counted, not hidden.
    backlog = max(0.0, reviews - a.review_capacity_per_day)

    missed = n_mal * (
        rates.malicious[ALLOW] + rates.malicious[REVIEW] * (1 - a.analyst_detection_rate)
    )
    false_blocks = n_ben * rates.benign[BLOCK]
    auto_blocked_malicious = n_mal * rates.malicious[BLOCK]

    cost = (
        reviews * a.cost_per_review
        + missed * a.cost_per_missed_malicious
        + false_blocks * a.cost_per_false_block
    )
    return {
        "reviews_per_day": reviews, "capacity_used": reviews / a.review_capacity_per_day,
        "backlog_per_day": backlog,
        "feasible": backlog == 0, "missed_malicious_per_day": missed,
        "false_blocks_per_day": false_blocks,
        "auto_blocked_malicious_per_day": auto_blocked_malicious,
        "expected_cost_per_day": cost, "review_cost_per_day": reviews * a.cost_per_review,
        "miss_cost_per_day": missed * a.cost_per_missed_malicious,
        "false_block_cost_per_day": false_blocks * a.cost_per_false_block,
    }


def candidate_thresholds(probabilities, max_points: int = 400) -> np.ndarray:
    # The calibrated scores bunch up near 0 and 1, so a plain linspace would waste most of its
    # points. Use the observed scores plus a few round values.
    p = np.unique(np.round(np.asarray(probabilities, dtype=float), 6))
    extra = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    grid = np.unique(np.concatenate([p, extra]))
    if len(grid) > max_points:
        grid = np.unique(np.quantile(grid, np.linspace(0, 1, max_points)))
    return grid


def choose_thresholds(
    probabilities, y, prevalence: float, a: Assumptions, max_points: int = 400
) -> tuple[ThresholdPolicy, dict[str, float]]:
    """Cheapest feasible (t_low, t_high) on labelled data. Falls back to the least-overloaded
    pair if nothing fits within capacity."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y)
    grid = candidate_thresholds(p, max_points)

    best, best_key = None, None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            policy = ThresholdPolicy(float(lo), float(hi))
            rates = TierRates.from_tiers(policy.decide(p), y)
            out = expected_outcome(rates, prevalence, a)
            # feasible beats infeasible; then cheaper; then less backlog
            key = (not out["feasible"], out["expected_cost_per_day"], out["backlog_per_day"])
            if best_key is None or key < best_key:
                best, best_key = (policy, out), key
    return best
