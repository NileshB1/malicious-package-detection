"""Compare the triage policy with the alternatives a security team actually has.

Thresholds are chosen on the validation split; every number reported here is measured on the test split and then scaled to a request stream
with a realistic share of malicious packages.
These are offline estimates under stated assumptions, not observed savings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pkgtriage.business.costs import SENSITIVITY_RANGES, Assumptions
from pkgtriage.business.policy import (
    ALLOW,
    REVIEW,
    ThresholdPolicy,
    TierRates,
    choose_thresholds,
    expected_outcome,
)

PER = 10_000  # report counts per 10,000 requests
# so scenarios with different volumes


def policy_rates(test_y, model_p, rule_flags, policy: ThresholdPolicy) -> dict[str, TierRates]:
    rule_tiers = np.where(np.asarray(rule_flags) == 1, REVIEW, ALLOW)
    return {
        "allow everything": TierRates.everyone(ALLOW),
        "review everything": TierRates.everyone(REVIEW),
        "rule screen (flag to review)": TierRates.from_tiers(rule_tiers, test_y),
        "model triage": TierRates.from_tiers(policy.decide(model_p), test_y),
    }


def _row(name: str, prevalence_name: str, prevalence: float, out: dict, a: Assumptions) -> dict:
    scale = PER / a.requests_per_day
    return {
        "scenario": prevalence_name,  "prevalence": prevalence,
        "policy": name,  "reviews_per_day": out["reviews_per_day"],
        "capacity_used": out["capacity_used"],  "feasible": out["feasible"],
        # staff needed to keep up with the reviews this policy generates
        "analysts_needed": out["reviews_per_day"] / a.reviews_per_analyst_per_day,
        "missed_malicious_per_10k": out["missed_malicious_per_day"] * scale,
        "false_blocks_per_10k": out["false_blocks_per_day"] * scale,
        "cost_per_10k": out["expected_cost_per_day"] * scale,  "review_cost_per_10k": out["review_cost_per_day"] * scale,
        "miss_cost_per_10k": out["miss_cost_per_day"] * scale,
        "false_block_cost_per_10k": out["false_block_cost_per_day"] * scale,
    }


def compare_policies(
    test_y, test_p,
    rule_flags,  policy: ThresholdPolicy,
    prevalence: float,  a: Assumptions, scenario: str = "custom"
) -> pd.DataFrame:
    """All four policies for one prevalence and one set of assumptions (used by the app)."""
    rates = policy_rates(test_y, test_p, rule_flags, policy)
    return pd.DataFrame(
        [
            _row(name, scenario, prevalence, expected_outcome(r, prevalence, a), a)
            for name, r in rates.items()
        ]
    )


def run_scenarios(val_p, val_y, test_p, test_y, rule_flags, scenarios: dict, a: Assumptions):
    """Returns (scenario table, threshold table)."""
    rows, thresholds = [], []
    for name, prevalence in scenarios.items():
        policy, val_out = choose_thresholds(val_p, val_y, prevalence, a)
        thresholds.append(
            {
                "scenario": name,  "prevalence": prevalence,
                "t_low": policy.t_low, "t_high": policy.t_high,
                "validation_cost_per_day": val_out["expected_cost_per_day"],
                "validation_feasible": val_out["feasible"],
            }
        )
        for pname, rates in policy_rates(test_y, test_p, rule_flags, policy).items():
            rows.append(_row(pname, name, prevalence, expected_outcome(rates, prevalence, a), a))
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def bootstrap_policy_cost(
    test_p,  test_y,
    policy: ThresholdPolicy, prevalence: float,
    a: Assumptions,  n_boot: int = 1000,
    seed: int = 42
) -> dict[str, float]:
    """Interval on the model policy's cost and misses from resampling the test packages."""
    rng = np.random.default_rng(seed)
    p, y = np.asarray(test_p), np.asarray(test_y)
    tiers = policy.decide(p)
    scale = PER / a.requests_per_day
    costs, misses = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out = expected_outcome(TierRates.from_tiers(tiers[idx], y[idx]), prevalence, a)
        costs.append(out["expected_cost_per_day"] * scale)
        misses.append(out["missed_malicious_per_day"] * scale)
    lo_c, hi_c = np.percentile(costs, [2.5, 97.5])
    lo_m, hi_m = np.percentile(misses, [2.5, 97.5])
    return {
        "cost_ci_low": lo_c, "cost_ci_high": hi_c,
        "missed_ci_low": lo_m,   "missed_ci_high": hi_m
    }


# The comparison that matters: a cautious team reviewing every new package, 
# staffed to keep up. 
# Its review labour is already in its cost, so it is compared regardless of capacity.
# "Allow everything" is not a serious comparator and would make any policy look brilliant.
REFERENCE_POLICY = "review everything"


def sensitivity(
    test_p, test_y, rule_flags, policy, prevalence, a: Assumptions, against: str = REFERENCE_POLICY
) -> pd.DataFrame:
    """One assumption at a time: how does the model policy's saving against against change?
    Can go negative. Thresholds stay fixed at the base-case choice - a team wouldn't retune
    every time an estimate moved.
    """
    rates = policy_rates(test_y, test_p, rule_flags, policy)

    def saving(assume: Assumptions) -> float:
        model_cost = expected_outcome(rates["model triage"], prevalence, assume)
        other_cost = expected_outcome(rates[against], prevalence, assume)
        
        diff = other_cost["expected_cost_per_day"] - model_cost["expected_cost_per_day"]
        return diff * PER / assume.requests_per_day

    base = saving(a)
    rows = []
    for name, (low, high) in SENSITIVITY_RANGES.items():
        rows.append(
            {
                "assumption": name, "compared_with": against,
                "base_value": getattr(a, name), "low_value": low, "high_value": high,
                "saving_at_low": saving(a.with_changes(**{name: low})),
              "saving_at_high": saving(a.with_changes(**{name: high})),
              "saving_at_base": base
            }
        )
    out = pd.DataFrame(rows)
    out["swing"] = (out.saving_at_high - out.saving_at_low).abs()
    return out.sort_values("swing", ascending=False).reset_index(drop=True)


def break_even_breach_probability(
    test_p, test_y, rule_flags, policy, prevalence, a
) -> pd.DataFrame:
    """Breach probability at which the model policy costs the same as each alternative.

    Cost is linear in breach probability, so two evaluations are enough to solve for it.
    """
    rates = policy_rates(test_y, test_p, rule_flags, policy)

    def costs(prob):
        assume = a.with_changes(breach_probability=prob)
        return {
            k: expected_outcome(r, prevalence, assume)["expected_cost_per_day"]
            for k, r in rates.items()
        }

    c0, c1 = costs(0.0), costs(1.0)
    rows = []
    for name in rates:
        if name == "model triage":
            continue
        # difference (other - model) as a line in breach probability
        d0 = c0[name] - c0["model triage"]

        d1 = c1[name] - c1["model triage"]
        slope = d1 - d0
        prob = -d0 / slope if slope != 0 else np.nan
        if 0 <= prob <= 1:
            verdict = f"model cheaper when breach probability is above {prob:.6f}"
        elif d0 >= 0 and d1 >= 0:
            verdict = "model cheaper at every breach probability"
        else:
            verdict = "alternative cheaper at every breach probability"
        
        rows.append({
                "compared_with": name,
                "break_even_breach_probability": prob if 0 <= prob <= 1 else np.nan,
             "verdict": verdict,
            } )
    return pd.DataFrame(rows)
