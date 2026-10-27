import numpy as np
import pytest

from pkgtriage.business.costs import load_assumptions
from pkgtriage.business.policy import (
    ALLOW,
    BLOCK,
    REVIEW,
    ThresholdPolicy,
    TierRates,
    choose_thresholds,
    expected_outcome,
)
from pkgtriage.business.simulate import break_even_breach_probability, compare_policies


class TestCosts:
    def test_unit_costs(self, simple_assumptions):
        a = simple_assumptions
        assert a.cost_per_review == 30
        assert a.cost_per_missed_malicious == pytest.approx(245_500)
        assert a.cost_per_false_block == 120
        assert a.review_capacity_per_day == 24

    def test_config_loads_with_sources(self):
        a, scenarios = load_assumptions()
        assert a.breach_cost == 4_910_000  # IBM 2025, the one sourced figure
        assert set(scenarios) == {"low", "base", "high"}
        assert all(0 < v < 1 for v in scenarios.values())


class TestPolicy:
    def test_decide_including_boundaries(self):
        policy = ThresholdPolicy(0.2, 0.8)
        got = policy.decide([0.1, 0.2, 0.5, 0.8, 0.9])
        assert list(got) == [ALLOW, REVIEW, REVIEW, BLOCK, BLOCK]

    @pytest.mark.parametrize("lo, hi", [(0.5, 0.4), (-0.1, 0.5), (0.2, 1.1)])
    def test_bad_thresholds(self, lo, hi):
        with pytest.raises(ValueError):
            ThresholdPolicy(lo, hi)

    def test_tier_rates(self):
        tiers = np.array([ALLOW, BLOCK, BLOCK, ALLOW, REVIEW])
        y = np.array([0, 1, 0, 0, 1])
        r = TierRates.from_tiers(tiers, y)
        assert r.malicious == {ALLOW: 0.0, REVIEW: 0.5, BLOCK: 0.5}
        assert r.benign[BLOCK] == pytest.approx(1 / 3)

    def test_allow_everything_by_hand(self, simple_assumptions):
        out = expected_outcome(TierRates.everyone(ALLOW), 0.01, simple_assumptions)
        # 200 requests, 1% malicious -> 2 slip through at 245,500 each
        assert out["missed_malicious_per_day"] == pytest.approx(2)
        assert out["expected_cost_per_day"] == pytest.approx(491_000)
        assert out["reviews_per_day"] == 0
        assert out["feasible"]

    def test_review_everything_by_hand(self, simple_assumptions):
        out = expected_outcome(TierRates.everyone(REVIEW), 0.01, simple_assumptions)
        # 200 reviews x $30, plus 10% of 2 malicious missed by the analyst
        assert out["reviews_per_day"] == pytest.approx(200)
        assert out["missed_malicious_per_day"] == pytest.approx(0.2)
        assert out["expected_cost_per_day"] == pytest.approx(6_000 + 0.2 * 245_500)
        assert out["backlog_per_day"] == pytest.approx(176)
        assert not out["feasible"]

    def test_choose_thresholds_on_separable_scores(self, simple_assumptions):
        rng = np.random.default_rng(1)
        y = np.r_[np.zeros(300), np.ones(300)]
        p = np.r_[rng.uniform(0, 0.3, 300), rng.uniform(0.7, 1, 300)]
        policy, out = choose_thresholds(p, y, 0.01, simple_assumptions, max_points=60)
        assert policy.t_low <= policy.t_high
        assert out["feasible"]
        assert out["missed_malicious_per_day"] == 0
        assert out["false_blocks_per_day"] == 0


class TestSimulate:
    def test_four_policies_compared(self, simple_assumptions):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.02, 0.99, 0.98])
        table = compare_policies(
            y, p, np.zeros(4), ThresholdPolicy(0.1, 0.9), 0.01, simple_assumptions
        )
        assert len(table) == 4
        model = table.set_index("policy").loc["model triage"]
        assert model.missed_malicious_per_10k == 0
        # a rule screen that never fires behaves exactly like allowing everything
        costs = table.set_index("policy").cost_per_10k
        assert costs["rule screen (flag to review)"] == pytest.approx(costs["allow everything"])

    def test_break_even_against_allow_everything(self, simple_assumptions):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.95, 0.99, 0.98])  # one safe package gets blocked
        table = break_even_breach_probability(
            p, y, np.zeros(4), ThresholdPolicy(0.1, 0.9), 0.01, simple_assumptions
        )
        row = table.set_index("compared_with").loc["allow everything"]
        assert 0 < row.break_even_breach_probability < 1
