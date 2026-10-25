"""Validate the raw data, run the leakage/artefact audit and draw the EDA figures

Outputs:
  reports/tables/*.csv   audit tables
    images/eda_*.png    figures for the report
"""

import logging

from pkgtriage import plots
from pkgtriage.config import load_config
from pkgtriage.data.audit import run_audit
from pkgtriage.data.download import load_raw
from pkgtriage.data.schema import COUNT_COLUMNS
from pkgtriage.data.validate import validate_raw

KEY_FEATURES = [
    "Network_Operations",
    "Home_DIR_Installation",
    "Process_Management_Operations",
    "Root_DIR_Installation",
    "Etc_DIR_Installation",
    "Remote_IP_Address_Access",
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    df = load_raw(cfg)

    report = validate_raw(df, cfg)
    print(report.summary())

    tables = run_audit(df)
    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(
            cfg.paths.tables_dir / f"{name}.csv",
            index=name in {"name_format_by_class", "family_sizes"},
        )

    print("\n", tables["name_format_by_class"], "\n")
    print(tables["family_stats"].T.to_string(header=False))

    plots.use_report_style()
    saved = [
        plots.save(
            plots.missingness_figure(tables["missingness_by_class"]), "eda_missingness_by_class"
        ),
        plots.save(plots.count_distributions_figure(df, KEY_FEATURES), "eda_count_distributions"),
        plots.save(
            plots.single_feature_auc_figure(tables["single_feature_auc"]), "eda_single_feature_auc"
        ),
        plots.save(
            plots.process_tokens_figure(tables["process_tokens_by_class"]),
            "eda_background_processes",
        ),
        plots.save(plots.family_size_figure(tables["family_sizes"]), "eda_family_sizes"),
        plots.save(plots.correlation_figure(df, COUNT_COLUMNS), "eda_correlation"),
    ]
    for p in saved:
        print("saved", p.relative_to(cfg.paths.images_dir.parent))


if __name__ == "__main__":
    main()
