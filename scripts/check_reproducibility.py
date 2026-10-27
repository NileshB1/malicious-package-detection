"""Check that the saved results can be rebuilt exactly from the pinned data and the code.

    python scripts/check_reproducibility.py

Writes reports/tables/reproducibility.csv. Exits with status 1 if any check fails.
"""

import sys

import numpy as np
import pandas as pd

from pkgtriage.config import load_config
from pkgtriage.data.download import load_raw, raw_csv_path, sha256_of
from pkgtriage.data.split import grouped_split
from pkgtriage.features.build import FEATURE_COLUMNS, NAME, build_features
from pkgtriage.logging_config import run, setup_logging
from pkgtriage.models import train
from pkgtriage.models.artifacts import load_bundle
from pkgtriage.models.dataset import load_splits


def main():
    log = setup_logging("check_reproducibility")
    log.info("checking reproducibility")
    cfg = load_config()
    checks = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    digest = sha256_of(raw_csv_path(cfg))
    record("raw data matches pinned checksum", digest == cfg.dataset.sha256, digest[:12])

    raw = load_raw(cfg)
    rebuilt = build_features(raw)
    saved = pd.read_parquet(cfg.paths.processed_dir / "features.parquet")
    same_features = np.array_equal(rebuilt.to_numpy(), saved[FEATURE_COLUMNS].to_numpy())
    record("feature table rebuilds identically", same_features, f"{rebuilt.shape}")

    assign = grouped_split(raw[NAME], raw["Level"], cfg.split, cfg.seed)
    saved_split = pd.read_csv(cfg.paths.splits_dir / "grouped.csv")
    same_split = assign.reset_index(drop=True)[["Package_Name", "split"]].equals(
        saved_split[["Package_Name", "split"]]
    )
    record("grouped split regenerates identically", same_split)

    bundle = load_bundle(cfg)
    splits = load_splits("grouped", cfg)
    test_X = splits["test"].X
    first = train.fit(splits["train"], bundle.params, cfg.seed).predict_proba(test_X)[:, 1]
    second = train.fit(splits["train"], bundle.params, cfg.seed).predict_proba(test_X)[:, 1]
    record("two training runs give identical predictions", np.array_equal(first, second))

    saved_pred = bundle.base_model.predict_proba(test_X)[:, 1]
    gap = float(np.max(np.abs(first - saved_pred)))
    record("retrained model matches the saved model", gap < 1e-9, f"max difference {gap:.2e}")

    out = pd.DataFrame(checks)
    out.to_csv(cfg.paths.tables_dir / "reproducibility.csv", index=False)
    sys.exit(0 if out.passed.all() else 1)


if __name__ == "__main__":
    run(main, "check_reproducibility")
