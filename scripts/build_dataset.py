"""Build the modelling table and the train/validation/test splits.

Outputs:
    data/processed/features.parquet   numeric features + Package_Name + Level
    data/splits/grouped.csv           family-grouped split (used for all reported results)
    data/splits/random.csv            stratified random split (comparison only)
    reports/tables/split_summary.csv
    reports/tables/feature_summary.csv
    images/split_family_overlap.png
"""

import logging

import pandas as pd

from pkgtriage import plots
from pkgtriage.config import load_config
from pkgtriage.data.download import load_raw
from pkgtriage.data.split import check_split, grouped_split, random_split
from pkgtriage.data.validate import validate_raw
from pkgtriage.features.build import FEATURE_GROUPS, LABEL, NAME, build_features


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    raw = load_raw(cfg)
    validate_raw(raw, cfg)

    features = build_features(raw)
    table = pd.concat([raw[[NAME]], features, raw[[LABEL]]], axis=1)

    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.processed_dir / "features.parquet"
    table.to_parquet(out, index=False)
    print(f"features: {features.shape[1]} columns, {len(table)} rows -> {out}")

    names, labels = raw[NAME], raw[LABEL]
    grouped = grouped_split(names, labels, cfg.split, cfg.seed)
    random = random_split(names, labels, cfg.split, cfg.seed)

    cfg.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(cfg.paths.splits_dir / "grouped.csv", index=False)
    random.to_csv(cfg.paths.splits_dir / "random.csv", index=False)

    summaries = []
    for kind, assign in (("grouped", grouped), ("random", random)):
        s = check_split(assign, labels)
        s.insert(0, "split_type", kind)
        summaries.append(s)
        print(f"\n{kind} split\n{s.round(3).to_string()}")

    leaks = summaries[0]["families_shared_with_other_splits"].sum()
    if leaks:
        raise RuntimeError(f"grouped split leaks {leaks} rows across splits")

    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    split_summary = pd.concat(summaries)
    split_summary.to_csv(cfg.paths.tables_dir / "split_summary.csv")

    plots.use_report_style()
    plots.save(plots.split_overlap_figure(split_summary.reset_index()), "split_family_overlap")

    group_of = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}
    desc = features.describe().T[["mean", "50%", "max"]]
    desc["nonzero_share"] = (features > 0).mean()
    desc.insert(0, "group", desc.index.map(group_of))
    desc.to_csv(cfg.paths.tables_dir / "feature_summary.csv")


if __name__ == "__main__":
    main()
