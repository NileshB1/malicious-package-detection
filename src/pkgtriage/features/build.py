from __future__ import annotations

import pandas as pd

from pkgtriage.data.schema import (
    DEPENDENCY_TEXT,  FILESYSTEM_COUNTS,
    NETWORK_COUNTS, PATTERN_TEXT,  PROCESS_TEXT,
    SYSCALL_COUNTS
)
from pkgtriage.features.parsers import (
    DEPENDENCY_FEATURES, PATTERN_FEATURES, PROCESS_FEATURES,
    TCP_FEATURES,
    parse_dependencies, parse_patterns, parse_processes, parse_tcp_transitions
)

LABEL = "Level"
NAME = "Package_Name"

# Network_Operations is a syscall count but it moves with network activity, so it sits with
# the network group. That matters for the ablations: "drop network" has to drop all of it.
NETWORK_GROUP = NETWORK_COUNTS + ["Network_Operations"] + TCP_FEATURES
SYSCALL_GROUP = [c for c in SYSCALL_COUNTS if c != "Network_Operations"]
PATTERN_GROUP = [f for f in PATTERN_FEATURES if f != "pat_missing_slots"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "filesystem": FILESYSTEM_COUNTS,
    "network": NETWORK_GROUP, "syscall": SYSCALL_GROUP + PATTERN_GROUP,
    "process": PROCESS_FEATURES,
    "dependency": [f for f in DEPENDENCY_FEATURES if not f.endswith("_missing")],
    "missingness": [f for f in DEPENDENCY_FEATURES if f.endswith("_missing")]
    + ["pat_missing_slots", "proc_lists_missing"],
}

FEATURE_COLUMNS = [f for group in FEATURE_GROUPS.values() for f in group]

# Groups whose values depend on how a package was installed (local archive or PyPI download),
# not only on what the package does. The Phase 3 audit is the evidence for this list.
COLLECTION_SENSITIVE = ["network", "dependency", "missingness"]


def _parse_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.itertuples(index=False):
        row = {}
        row.update(parse_tcp_transitions(rec.State_Transition))
        row.update(parse_processes(*(getattr(rec, c) for c in PROCESS_TEXT)))
        row.update(parse_dependencies(*(getattr(rec, c) for c in DEPENDENCY_TEXT)))
        row.update(parse_patterns(*(getattr(rec, c) for c in PATTERN_TEXT)))
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature table, one row per package, in FEATURE_COLUMNS order.

    Package_Name and the label are not used here; the caller keeps them alongside.
    """
    parsed = _parse_rows(df)
    counts = df[FILESYSTEM_COUNTS + NETWORK_COUNTS + SYSCALL_COUNTS].astype(float)

    # all five process columns go missing together, one flag is enough
    parsed["proc_lists_missing"] = df[PROCESS_TEXT].isna().all(axis=1).astype(float)

    features = pd.concat([counts, parsed], axis=1)
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise KeyError(f"feature builder did not produce: {missing}")
    return features[FEATURE_COLUMNS]


def columns_for(groups: list[str] | None = None, drop: list[str] | None = None) -> list[str]:
    """Feature columns for a set of groups, used by the ablation runs."""
    chosen = groups or list(FEATURE_GROUPS)
    chosen = [g for g in chosen if g not in (drop or [])]
    return [f for g in chosen for f in FEATURE_GROUPS[g]]
