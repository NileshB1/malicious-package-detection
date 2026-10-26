import streamlit as st

import components as ui
from pkgtriage import business_plots
from pkgtriage.business.policy import ThresholdPolicy, choose_thresholds
from pkgtriage.business.simulate import compare_policies

ui.sidebar()
base, scenarios = ui.assumptions()
scores = ui.test_scores()

st.title("Policy simulator")
st.write(
    "Change the assumptions and the thresholds, and see what each way of handling package "
    "requests would cost. Outcomes come from the held-out test packages, scaled to the request "
    "stream you set here."
)

with st.expander("Assumptions", expanded=True):
    c1, c2, c3 = st.columns(3)
    prevalence = c1.select_slider(
        "Share of requests that are malicious",
        options=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        value=scenarios["base"], format_func=lambda v: f"{v:.1%}",
    )
    requests = c1.number_input("Requests per day", 10, 5000, int(base.requests_per_day), step=10)
    analysts = c2.number_input("Analysts", 1, 50, int(base.analysts))
    per_analyst = c2.number_input(
        "Reviews per analyst per day", 1, 100, int(base.reviews_per_analyst_per_day)
    )
    detection = c2.slider("Analyst detection rate", 0.5, 1.0, base.analyst_detection_rate, 0.05)
    minutes = c3.number_input("Minutes per review", 5, 240, int(base.review_minutes), step=5)
    rate = c3.number_input("Analyst hourly cost (USD)", 10, 300, int(base.analyst_hourly_rate))
    breach_p = c1.select_slider(
        "Chance a malicious install becomes a breach",
        options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5],
        value=base.breach_probability, format_func=lambda v: f"{v:.1%}",
    )
    breach_cost = c3.number_input(
        "Cost of a breach (USD)", 100_000, 50_000_000, int(base.breach_cost), step=100_000,
        help="Default: IBM Cost of a Data Breach 2025, supply chain compromise average.",
    )
    dev_hours = c3.number_input(
        "Developer hours lost per wrong block", 0.0, 40.0, base.developer_hours_per_false_block
    )

a = base.with_changes(
    requests_per_day=float(requests), analysts=float(analysts),
    reviews_per_analyst_per_day=float(per_analyst), analyst_detection_rate=float(detection),
    review_minutes=float(minutes), analyst_hourly_rate=float(rate),
    breach_probability=float(breach_p), breach_cost=float(breach_cost),
    developer_hours_per_false_block=float(dev_hours),
)

st.subheader("Thresholds")
policy = ui.current_policy()
c1, c2 = st.columns(2)
t_low, t_high = c1.slider(
    "Allow below / block at or above", 0.0, 1.0, (policy.t_low, policy.t_high), 0.001,
    format="%.3f",
)
if c2.button("Find the cheapest thresholds for these assumptions"):
    with st.spinner("Searching on the validation packages..."):
        best, _ = choose_thresholds(scores["val_p"], scores["val_y"], prevalence, a, max_points=120)
    t_low, t_high = best.t_low, best.t_high
    c2.success(f"Allow below {t_low:.3f}, block at or above {t_high:.3f}")
st.session_state.t_low, st.session_state.t_high = float(t_low), float(t_high)
c2.caption("These thresholds also apply on the Review queue page.")

table = compare_policies(
    scores["test_y"], scores["test_p"], scores["rule_flags"],
    ThresholdPolicy(t_low, t_high), prevalence, a,
)

view = table.copy()
view["policy"] = view.policy.map(business_plots.POLICY_LABELS)
view["fits capacity"] = view.feasible.map({True: "yes", False: "no"})
st.dataframe(
    view[["policy", "reviews_per_day", "analysts_needed", "fits capacity",
          "missed_malicious_per_10k", "false_blocks_per_10k", "cost_per_10k"]],
    width="stretch", hide_index=True,
    column_config={
        "reviews_per_day": st.column_config.NumberColumn("Reviews / day", format="%.1f"),
        "analysts_needed": st.column_config.NumberColumn("Analysts needed", format="%.1f"),
        "missed_malicious_per_10k": st.column_config.NumberColumn(
            "Malicious let through / 10k", format="%.2f"),
        "false_blocks_per_10k": st.column_config.NumberColumn(
            "Safe blocked / 10k", format="%.0f"),
        "cost_per_10k": st.column_config.NumberColumn("Expected cost / 10k (USD)", format="%.0f"),
    },
)

ui.report_style()
c1, c2 = st.columns(2)
with c1:
    ui.pyplot(business_plots.cost_by_scenario_figure(table.assign(scenario="Your settings")))
with c2:
    ui.pyplot(business_plots.workload_figure(table, a.review_capacity_per_day))

st.caption(
    "Expected cost = review time + missed malicious packages x breach chance x breach cost + "
    "developer time lost to wrong blocks. Offline estimate under these assumptions."
)
