import math

import pytest

from pkgtriage.features.parsers import (
    DEPENDENCY_FEATURES,
    PATTERN_FEATURES,
    PROCESS_FEATURES,
    TCP_FEATURES,
    parse_dependencies,
    parse_patterns,
    parse_processes,
    parse_tcp_transitions,
)

NAN = float("nan")


class TestTcp:
    def test_counts_states_ports_and_raw_ips(self):
        cell = (
            "{'SYN_SENT -> ->': 3, '443 -> ESTABLISHED': 2, '8889 -> SYN_SENT': 1, "
            "'142.250.194.78 -> 443': 4, 'LISTEN -> ->': 1}"
        )
        out = parse_tcp_transitions(cell)
        assert out["tcp_events"] == 11
        assert out["tcp_syn_sent"] == 4  # plain state plus the one on port 8889
        assert out["tcp_established"] == 2
        assert out["tcp_listen"] == 1
        assert out["tcp_port_443"] == 2
        assert out["tcp_port_other"] == 1
        assert out["tcp_distinct_ports"] == 2
        assert out["tcp_raw_ip_events"] == 4

    @pytest.mark.parametrize("cell", [NAN, None, "", "{broken", "[1, 2]", "{'x -> ->': 'many'}"])
    def test_bad_cells_give_zeros_not_errors(self, cell):
        out = parse_tcp_transitions(cell)
        assert set(out) == set(TCP_FEATURES)
        assert all(v == 0 for v in out.values())


class TestProcesses:
    def test_environment_tokens_are_ignored(self):
        out = parse_processes("pip, opensnoop-bpfcc, tracker-miner-f, strace", "pip3, chrome")
        assert out == dict.fromkeys(PROCESS_FEATURES, 0.0)

    def test_buckets(self):
        out = parse_processes("cc1, ld, bash", "curl, rootkit", NAN)
        assert out["proc_build_tools"] == 2
        assert out["proc_shell_tools"] == 2
        assert out["proc_other_distinct"] == 1

    def test_same_process_in_several_lists_counts_once(self):
        out = parse_processes("bash", "bash", "bash")
        assert out["proc_shell_tools"] == 1


class TestDependencies:
    def test_counts_and_missing_flags(self):
        out = parse_dependencies("a-1; b-2;", NAN, "c>=1;")
        assert out["dep_total_count"] == 2
        assert out["dep_direct_count"] == 0
        assert out["dep_indirect_count"] == 1
        assert out["dep_direct_missing"] == 1
        assert out["dep_total_missing"] == 0
        assert set(out) == set(DEPENDENCY_FEATURES)

    def test_trailing_semicolons_and_spaces(self):
        assert parse_dependencies("a; ; b ;;", "x", "y")["dep_total_count"] == 2


class TestPatterns:
    def test_pools_tokens_across_slots(self):
        out = parse_patterns(
            "newfstatat -> openat -> fstat",
            "socket -> connect -> error=ECONNREFUSED -> fd=3",
            "syscall_0x1000 -> read -> read -> no-error -> no-fd",
            None,
            NAN,
        )
        assert out["pat_file"] == 5
        assert out["pat_network"] == 2
        assert out["pat_errors"] == 1
        assert out["pat_unknown_syscalls"] == 1
        assert out["pat_missing_slots"] == 2
        # newfstatat, openat, fstat, socket, connect, error=..., syscall_0x1000, read
        assert out["pat_distinct_tokens"] == 8

    def test_all_missing(self):
        out = parse_patterns(*[NAN] * 10)
        assert out["pat_missing_slots"] == 10
        assert set(out) == set(PATTERN_FEATURES)
        assert not any(math.isnan(v) for v in out.values())
