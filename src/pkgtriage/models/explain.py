"""SHAP explanations, phrased for an analyst rather than a data scientist.

SHAP runs on the uncalibrated boosted model. Calibration is a monotonic squeeze of its
output, so the ranking of reasons doesn't change, and TreeExplainer is exact on the raw
model (values add up to its log-odds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from pkgtriage.features.build import COLLECTION_SENSITIVE, FEATURE_GROUPS

# What each feature means in words an analyst would use. Values are counts during install.
FEATURE_TEXT = {
  "Root_DIR_Installation": "file accesses under /root",
    "Temporary_DIR_Installation": "file accesses under /tmp",
    "Home_DIR_Installation": "file accesses in home directories",
  "User_Access": "file accesses under /usr",
  "Sys_Access": "file accesses under /sys",
   "Etc_DIR_Installation": "file accesses under /etc",
    "Other_DIR_Installation": "file accesses elsewhere (e.g. /proc)",
  "Local_IP_Address_Access": "local IP addresses used",
  "Remote_IP_Address_Access": "remote IP addresses contacted",
  "Local_Port_Access": "local ports opened",
    "Remote_Port_Access": "remote ports contacted",
    "Network_Operations": "network system calls",
  "tcp_events": "TCP state changes",
  "tcp_established": "TCP connections established",
  "tcp_syn_sent": "outgoing connection attempts",
  "tcp_close": "TCP connections closed",
    "tcp_close_wait": "connections left waiting to close",    "tcp_fin_wait1": "connections being shut down",
    "tcp_listen": "listening sockets opened", "tcp_port_443": "HTTPS (443) traffic events",
    "tcp_port_80": "plain HTTP (80) traffic events",
  "tcp_port_other": "traffic on unusual ports",
  "tcp_distinct_ports": "distinct remote ports", "tcp_raw_ip_events": "connections straight to an IP address",
    "File_Operations": "file system calls",
  "Process_Management_Operations": "process creation and control calls", 
  "IO_Operations": "general I/O calls",
    "Time_Operations": "timer calls",
    "Security_Operations": "permission and identity calls (chmod, setuid...)",
  "pat_file": "file calls in the common syscall sequences",
  "pat_network": "network calls in the common syscall sequences",
  "pat_process": "process calls in the common syscall sequences", "pat_memory": "memory calls in the common syscall sequences",
  "pat_io_wait": "wait/poll calls in the common syscall sequences",
  "pat_time": "timer calls in the common syscall sequences",
   "pat_errors": "errors in the common syscall sequences",
    "pat_unknown_syscalls": "unnamed system calls",
  "pat_distinct_tokens": "variety of system calls in the common sequences",
  "proc_build_tools": "compilers or build tools started",
  "proc_shell_tools": "shell or file utilities started",
  "proc_other_distinct": "other unexpected processes started",
  "dep_total_count": "dependencies resolved", "dep_direct_count": "declared dependencies",
    "dep_indirect_count": "indirect dependencies", "dep_total_missing": "no dependency record",
    "dep_direct_missing": "no declared dependencies recorded",
  "dep_indirect_missing": "no indirect dependencies recorded",
  "pat_missing_slots": "missing syscall sequences",
  "proc_lists_missing": "no process list recorded",
}

GROUP_OF = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}


@dataclass
class Driver:
    feature: str
    value: float
    typical_benign: float
    contribution: float  #positive pushes towards malicious

    @property
    def raises_risk(self) -> bool:
        return self.contribution > 0

    @property
    def collection_sensitive(self) -> bool:
        # partly reflects how the dataset was collected, so weaker evidence for an analyst
        return GROUP_OF.get(self.feature) in COLLECTION_SENSITIVE

    def sentence(self) -> str:
        what = FEATURE_TEXT.get(self.feature, self.feature)
        if self.feature.endswith("_missing") or self.feature == "proc_lists_missing":
            state = "yes" if self.value else "no"
            return f"{what}: {state}"
        return f"{what}: {self.value:,.0f} (typical benign install: {self.typical_benign:,.0f})"


class Explainer:
    def __init__(self, base_model, benign_medians: pd.Series):
        self.model = base_model
        self.columns = list(base_model.feature_names_in_)
        self.benign_medians = benign_medians.reindex(self.columns)
        self._shap = shap.TreeExplainer(base_model)

    def values(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns]
        return pd.DataFrame(self._shap.shap_values(X), columns=self.columns, index=X.index)

    def top_drivers(self, row: pd.Series, k: int = 3) -> list[Driver]:
        contrib = self.values(row.to_frame().T).iloc[0]
        order = contrib.abs().sort_values(ascending=False).index[:k]
        return [
            Driver(
                feature=f,    value=float(row[f]),
                typical_benign=float(self.benign_medians[f]),
                contribution=float(contrib[f])
            )
            for f in order
        ]


def global_importance(shap_values: pd.DataFrame) -> pd.DataFrame:
    imp = shap_values.abs().mean().rename("mean_abs_shap").to_frame()
    imp["group"] = imp.index.map(GROUP_OF)
    imp["description"] = imp.index.map(FEATURE_TEXT)
    return imp.sort_values("mean_abs_shap", ascending=False)


def group_importance(shap_values: pd.DataFrame) -> pd.Series:
    by_group = shap_values.abs().T.groupby(GROUP_OF).sum().T.mean()
    return (by_group / by_group.sum()).sort_values(ascending=False).rename("share")


def benign_medians(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    return X.loc[np.asarray(y) == 0].median()
