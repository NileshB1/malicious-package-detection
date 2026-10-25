"""Business assumptions and the unit costs derived from them.

All money is in USD, because the one sourced figure (IBM 2025) is in USD.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from pkgtriage.config import load_business_assumptions


@dataclass(frozen=True)
class Assumptions:
    requests_per_day: float
    analysts: float
    reviews_per_analyst_per_day: float
    analyst_detection_rate: float
    breach_cost: float
    breach_probability: float
    analyst_hourly_rate: float
    review_minutes: float
    developer_hourly_rate: float
    developer_hours_per_false_block: float

    @property
    def review_capacity_per_day(self) -> float:
        return self.analysts * self.reviews_per_analyst_per_day

    @property
    def cost_per_review(self) -> float:
        return self.review_minutes / 60 * self.analyst_hourly_rate

    @property
    def cost_per_missed_malicious(self) -> float:
        # expected, not certain: most malicious installs never become a reported breach
        return self.breach_probability * self.breach_cost

    @property
    def cost_per_false_block(self) -> float:
        return self.developer_hours_per_false_block * self.developer_hourly_rate

    def with_changes(self, **changes) -> Assumptions:
        return replace(self, **changes)

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _value(entry):
    # entries in business.yaml are either plain numbers or {value: ..., source: ...}
    return float(entry["value"] if isinstance(entry, dict) else entry)


def load_assumptions(path=None) -> tuple[Assumptions, dict[str, float]]:
    """Returns the assumptions and the prevalence scenarios from configs/business.yaml."""
    raw = load_business_assumptions(path)
    cap, costs = raw["capacity"], raw["costs"]
    assumptions = Assumptions(
        requests_per_day=_value(raw["requests_per_day"]),
        analysts=_value(cap["analysts"]),
        reviews_per_analyst_per_day=_value(cap["reviews_per_analyst_per_day"]),
        analyst_detection_rate=_value(cap["analyst_detection_rate"]),
        breach_cost=_value(costs["supply_chain_breach_cost"]),
        breach_probability=_value(costs["breach_probability_if_malicious_installed"]),
        analyst_hourly_rate=_value(costs["analyst_hourly_rate"]),
        review_minutes=_value(costs["review_minutes_per_package"]),
        developer_hourly_rate=_value(costs["developer_hourly_rate"]),
        developer_hours_per_false_block=_value(costs["developer_hours_lost_per_false_block"]),
    )
    scenarios = {k: float(v) for k, v in raw["prevalence_scenarios"].items()}
    return assumptions, scenarios


# Ranges for the one-way sensitivity analysis: (low, high) for each assumption.
# Wide on purpose for the ones nobody publishes. Request volume and team size are left out:
# they decide whether a policy fits the team's capacity, not what it costs per request.
SENSITIVITY_RANGES = {
    "breach_probability": (0.01, 0.5),
    "breach_cost": (1_000_000, 10_000_000),
    "review_minutes": (10, 60),
    "analyst_hourly_rate": (35, 100),
    "analyst_detection_rate": (0.7, 1.0),
    "developer_hours_per_false_block": (0.5, 8),
}
