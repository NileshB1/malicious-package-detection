"""Turn the text columns of QUT-DV25 into numbers.

Every parser takes one cell and returns a flat dict of counts. Bad or empty cells give zeros
rather than raising, because a handful of malformed traces shouldn't stop a whole batch.
"""

from __future__ import annotations

import ast
import math
import re

from pkgtriage.data.schema import ENVIRONMENT_PROCESSES


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value) == ""


# TCP state transitions
#
# The cell is a dict literal such as {'SYN_SENT -> ->': 136, '443 -> ESTABLISHED': 22, ...}.
# Three kinds of keys turn up:
#   "STATE -> ->"        a state change with no port attached
#   "PORT -> STATE"      a state change on a remote port
#   "a.b.c.d -> PORT"    a connection to a raw IP address

_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
TCP_STATES = ["ESTABLISHED", "SYN_SENT", "CLOSE", "CLOSE_WAIT", "FIN_WAIT1", "LISTEN"]

TCP_FEATURES = [
    "tcp_events",
    *[f"tcp_{s.lower()}" for s in TCP_STATES],
    "tcp_port_443",
    "tcp_port_80",
    "tcp_port_other",
    "tcp_distinct_ports",
    "tcp_raw_ip_events",
]


def parse_tcp_transitions(cell) -> dict[str, float]:
    out = dict.fromkeys(TCP_FEATURES, 0.0)
    if _is_blank(cell):
        return out
    try:
        transitions = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return out
    if not isinstance(transitions, dict):
        return out

    ports = set()
    for key, count in transitions.items():
        try:
            count = float(count)
        except (TypeError, ValueError):
            continue
        left, _, right = (part.strip() for part in str(key).partition("->"))
        right = right.replace("->", "").strip()
        out["tcp_events"] += count

        if _IP.match(left):
            out["tcp_raw_ip_events"] += count
            continue

        state = left if not left.isdigit() else right
        if state in TCP_STATES:
            out[f"tcp_{state.lower()}"] += count

        if left.isdigit():
            ports.add(left)
            if left == "443":
                out["tcp_port_443"] += count
            elif left == "80":
                out["tcp_port_80"] += count
            else:
                # 8080, 8889, 0 ... anything that isn't plain web traffic
                out["tcp_port_other"] += count

    out["tcp_distinct_ports"] = float(len(ports))
    return out


# Process lists
#
# Comma-separated "top N" process names from filetop. Tokens that belong to the sandbox are
# dropped first (see data/schema.py). What is left is grouped into a few generic buckets;
# individual names are not used because picking them after looking at the labels would be
# fitting the features to the answer.

BUILD_TOOLS = {"cc1", "cc1plus", "ld", "ld.bfd", "as", "gcc", "g++", "rustc", "cargo", "make"}
SHELL_TOOLS = {
    "sh",
    "bash",
    "dash",
    "cat",
    "grep",
    "sed",
    "awk",
    "ps",
    "fuser",
    "tar",
    "gzip",
    "xz",
    "zstd",
    "cpio",
    "file",
    "curl",
    "wget",
}

PROCESS_FEATURES = ["proc_build_tools", "proc_shell_tools", "proc_other_distinct"]


def process_tokens(*cells) -> set[str]:
    tokens = set()
    for cell in cells:
        if _is_blank(cell):
            continue
        tokens.update(t.strip() for t in str(cell).split(",") if t.strip())
    return tokens - ENVIRONMENT_PROCESSES


def parse_processes(*cells) -> dict[str, float]:
    tokens = process_tokens(*cells)
    build = tokens & BUILD_TOOLS
    shell = tokens & SHELL_TOOLS
    return {
        "proc_build_tools": float(len(build)),
        "proc_shell_tools": float(len(shell)),
        "proc_other_distinct": float(len(tokens - build - shell)),
    }


# Dependency lists
#
# Semicolon-separated, e.g. "certifi>=2017.4.17; idna<4,>=2.5;". Only the counts are used;
# the names themselves would let the model memorise particular packages.

DEPENDENCY_FEATURES = [
    "dep_total_count",
    "dep_direct_count",
    "dep_indirect_count",
    "dep_total_missing",
    "dep_direct_missing",
    "dep_indirect_missing",
]


def count_items(cell) -> float:
    if _is_blank(cell):
        return 0.0
    return float(sum(1 for part in str(cell).split(";") if part.strip()))


def parse_dependencies(total, direct, indirect) -> dict[str, float]:
    return {
        "dep_total_count": count_items(total),
        "dep_direct_count": count_items(direct),
        "dep_indirect_count": count_items(indirect),
        "dep_total_missing": float(_is_blank(total)),
        "dep_direct_missing": float(_is_blank(direct)),
        "dep_indirect_missing": float(_is_blank(indirect)),
    }


# Syscall patterns
#
# Pattern_1..10 each hold one n-gram like "newfstatat -> openat -> fstat", and 6..10 carry
# extra "no-error" / "error=ENOENT" / "fd=3" markers. The slot a pattern sits in doesn't mean
# much on its own, so tokens from all ten are pooled and counted by syscall family.

SYSCALL_FAMILIES = {
    "file": {
        "newfstatat",
        "fstat",
        "openat",
        "lseek",
        "read",
        "write",
        "close",
        "fcntl",
        "getdents64",
        "unlinkat",
        "unlink",
        "mkdir",
        "rmdir",
        "chown",
        "utimensat",
        "getcwd",
        "umask",
    },
    "network": {
        "socket",
        "connect",
        "getsockname",
        "getpeername",
        "setsockopt",
        "recvfrom",
        "sendto",
    },
    "process": {
        "exit_group",
        "wait4",
        "getpid",
        "geteuid",
        "futex",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "set_robust_list",
        "rseq",
        "clock_nanosleep",
    },
    "memory": {"mmap", "munmap", "brk"},
    "io_wait": {"ioctl", "poll", "ppoll", "pselect6"},
    "time": {"clock_gettime"},
}

PATTERN_FEATURES = [
    *[f"pat_{family}" for family in SYSCALL_FAMILIES],
    "pat_errors",
    "pat_unknown_syscalls",
    "pat_distinct_tokens",
    "pat_missing_slots",
]


def parse_patterns(*cells) -> dict[str, float]:
    out = dict.fromkeys(PATTERN_FEATURES, 0.0)
    distinct = set()
    for cell in cells:
        if _is_blank(cell):
            out["pat_missing_slots"] += 1
            continue
        for token in (t.strip() for t in str(cell).split("->")):
            if not token or token in ("no-error", "no-fd") or token.startswith("fd="):
                continue
            distinct.add(token)
            if token.startswith("error="):
                out["pat_errors"] += 1
                continue
            for family, names in SYSCALL_FAMILIES.items():
                if token in names:
                    out[f"pat_{family}"] += 1
                    break
            else:
                # raw numbers like syscall_0x1000 that strace couldn't name
                out["pat_unknown_syscalls"] += 1
    out["pat_distinct_tokens"] = float(len(distinct))
    return out
