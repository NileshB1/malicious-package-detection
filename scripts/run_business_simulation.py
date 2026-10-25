"""Pick allow/review/block thresholds and estimate workload, 
misses and cost for each policy
"""

import json

import pandas as pd

from pkgtriage import business_plots, plots
from pkgtriage.business.costs import load_assumptions
from pkgtriage.business.policy import ThresholdPolicy
from pkgtriage.business.simulate import (
    bootstrap_policy_cost,
    break_even_breach_probability,
    run_scenarios,
    sensitivity,
)
from pkgtriage.config import load_config
from pkgtriage.models.artifacts import load_bundle
from pkgtriage.models.dataset import load_splits

BASE = "base"


def main():
    cfg = load_config()
    a, scenarios = load_assumptions()
    bundle = load_bundle(cfg)
    splits = load_splits("grouped", cfg)
    val, test = splits["validation"], splits["test"]

    val_p = bundle.predict_proba(val.X)
    test_p = bundle.predict_proba(test.X)
    rule_flags = bundle.baseline.flag(test.X)

    print(
        f"capacity {a.review_capacity_per_day:.0f} reviews/day, "
        f"{a.requests_per_day:.0f} requests/day"
    )
    print(
        f"unit costs: review ${a.cost_per_review:,.0f}, missed malicious "
        f"${a.cost_per_missed_malicious:,.0f}, false block ${a.cost_per_false_block:,.0f}\n"
    )

    table, thresholds = run_scenarios(val_p, val.y, test_p, test.y, rule_flags, scenarios, a)
    print(thresholds.to_string(index=False))
    cols = [
        "scenario",   "policy", "reviews_per_day",
        "feasible", "missed_malicious_per_10k",
        "false_blocks_per_10k","cost_per_10k"
    ]
    print("\n" + table[cols].round(2).to_string(index=False))

    base_row = thresholds[thresholds.scenario == BASE].iloc[0]
    policy = ThresholdPolicy(base_row.t_low, base_row.t_high)
    prevalence = scenarios[BASE]

    ci = bootstrap_policy_cost(test_p, test.y, policy, prevalence, a, seed=cfg.seed)
    print(
        "\nmodel triage, base scenario, 95% bootstrap interval:",
        {k: round(v, 2) for k, v in ci.items()},
    )

    sens = sensitivity(test_p, test.y, rule_flags, policy, prevalence, a)
    print("\n" + sens.round(1).to_string(index=False))

    breakeven = break_even_breach_probability(test_p, test.y, rule_flags, policy, prevalence, a)
    print("\n" + breakeven.to_string(index=False))

    tables = cfg.paths.tables_dir
    thresholds.to_csv(tables / "policy_thresholds.csv", index=False)
    table.to_csv(tables / "business_scenarios.csv", index=False)
    pd.DataFrame([ci]).to_csv(tables / "business_base_ci.csv", index=False)
    sens.to_csv(tables / "sensitivity.csv", index=False)
    breakeven.to_csv(tables / "break_even.csv", index=False)

    policy_file = cfg.paths.artifacts_dir / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "t_low": policy.t_low,  "t_high": policy.t_high,
                "scenario": BASE,  "prevalence": prevalence,
                "chosen_on": "validation split"
            },
            indent=2
        )
    )

    plots.use_report_style()
    plots.save(business_plots.cost_by_scenario_figure(table), "business_cost_by_scenario")
    base_table = table[table.scenario == BASE]
    plots.save(
        business_plots.workload_figure(base_table, a.review_capacity_per_day), "business_workload"
    )
    plots.save(business_plots.tornado_figure(sens), "business_sensitivity")
    print("\ntables in reports/tables, figures in images/, thresholds in", policy_file)

#main method
if __name__ == "__main__":
    main()

