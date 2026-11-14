"""Pick allow/review/block thresholds and estimate workload,
misses and cost for each policy

Runs twice: for the main model, and for the behaviour-only variant (same settings, without the
collection-sensitive feature groups), so the report can give both business cases.
"""

import json

import pandas as pd

from pkgtriage import business_plots, plots
from pkgtriage.business.costs import load_assumptions
from pkgtriage.business.policy import ThresholdPolicy
from pkgtriage.business.simulate import (
    base_rate_effect,
    bootstrap_policy_cost,
    break_even_breach_probability,
    run_scenarios,
    sensitivity,
)
from pkgtriage.config import load_config
from pkgtriage.logging_config import run, setup_logging
from pkgtriage.models.artifacts import BEHAVIOUR_BUNDLE_NAME, load_bundle
from pkgtriage.models.dataset import load_splits

BASE = "base"
MODEL = "model triage"


def variant_summary(variant: str, table: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: the model policy's outcome, and its saving over the alternatives."""
    rows = []
    for scenario, part in table.groupby("scenario", sort=False):
        by = part.set_index("policy")
        m = by.loc[MODEL]
        th = thresholds.set_index("scenario").loc[scenario]
        rows.append({
            "variant": variant, "scenario": scenario, "prevalence": m.prevalence,
            "t_low": th.t_low, "t_high": th.t_high,
            "reviews_per_day": m.reviews_per_day, "analysts_needed": m.analysts_needed,
            "feasible": m.feasible,
            "missed_malicious_per_10k": m.missed_malicious_per_10k,
            "false_blocks_per_10k": m.false_blocks_per_10k, "cost_per_10k": m.cost_per_10k,
            "block_precision": m.block_precision,
            "false_alarms_per_real_attack": m.false_alarms_per_real_attack,
            "saving_vs_review_everything": by.loc["review everything", "cost_per_10k"] - m.cost_per_10k,
            "saving_vs_rule_screen": by.loc["rule screen (flag to review)", "cost_per_10k"] - m.cost_per_10k,
        })
    return pd.DataFrame(rows)


def main():
    log = setup_logging("run_business_simulation")
    log.info("simulating policies")
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

    effect = base_rate_effect(test_p, test.y, policy, a).assign(variant="all features")

    # --- behaviour-only variant: the conservative business case --------------------------
    b_bundle = load_bundle(cfg, BEHAVIOUR_BUNDLE_NAME)
    b_val_p, b_test_p = b_bundle.predict_proba(val.X), b_bundle.predict_proba(test.X)
    b_table, b_thresholds = run_scenarios(
        val_p=b_val_p, val_y=val.y, test_p=b_test_p, test_y=test.y,
        rule_flags=rule_flags, scenarios=scenarios, a=a,
    )
    b_row = b_thresholds[b_thresholds.scenario == BASE].iloc[0]
    b_policy = ThresholdPolicy(b_row.t_low, b_row.t_high)
    effect = pd.concat([
        effect, base_rate_effect(b_test_p, test.y, b_policy, a).assign(variant="behaviour only"),
    ], ignore_index=True)
    variants = pd.concat([
        variant_summary("all features", table, thresholds),
        variant_summary("behaviour only", b_table, b_thresholds),
    ], ignore_index=True)

    print("\nbase-rate effect (thresholds fixed at the base case):")
    print(effect[["variant", "setting", "block_precision", "flag_precision",
                  "false_alarms_per_real_attack"]].round(3).to_string(index=False))
    print("\nmodel triage, all features vs behaviour only:")
    print(variants[["variant", "scenario", "t_low", "t_high", "reviews_per_day",
                    "missed_malicious_per_10k", "false_blocks_per_10k", "cost_per_10k",
                    "block_precision"]].round(3).to_string(index=False))

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
    effect.to_csv(tables / "base_rate_effect.csv", index=False)
    variants.to_csv(tables / "business_model_variants.csv", index=False)
    b_table.to_csv(tables / "business_scenarios_behaviour.csv", index=False)
    b_thresholds.to_csv(tables / "policy_thresholds_behaviour.csv", index=False)

    policy_file = cfg.paths.artifacts_dir / "policy.json"
    for path, pol in [(policy_file, policy),
                      (cfg.paths.artifacts_dir / "policy_behaviour.json", b_policy)]:
        path.write_text(
            json.dumps(
                {
                    "t_low": pol.t_low,  "t_high": pol.t_high,
                    "scenario": BASE,  "prevalence": prevalence, "chosen_on": "validation split"
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
    run(main, "run_business_simulation")

