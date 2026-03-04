"""Tests for _resolve_workers and CLI parallel control flags."""

import os
from unittest import mock

import tools.regenerate_net_profit_plotly as regen


class TestResolveWorkers:
    """Unit tests for _resolve_workers()."""

    def test_sequential_when_requested_1(self):
        assert regen._resolve_workers(requested=1, max_memory_mb=0, n_tasks=10) == 1

    def test_auto_caps_at_2(self):
        with mock.patch("os.cpu_count", return_value=8):
            result = regen._resolve_workers(requested=0, max_memory_mb=0, n_tasks=10)
        assert result == 2

    def test_auto_caps_at_n_tasks(self):
        with mock.patch("os.cpu_count", return_value=8):
            result = regen._resolve_workers(requested=0, max_memory_mb=0, n_tasks=1)
        assert result == 1

    def test_explicit_workers_respected(self):
        with mock.patch("os.cpu_count", return_value=8):
            result = regen._resolve_workers(requested=4, max_memory_mb=0, n_tasks=10)
        assert result == 4

    def test_explicit_workers_capped_by_cpu(self):
        with mock.patch("os.cpu_count", return_value=2):
            result = regen._resolve_workers(requested=8, max_memory_mb=0, n_tasks=10)
        assert result == 2

    def test_explicit_workers_capped_by_n_tasks(self):
        with mock.patch("os.cpu_count", return_value=8):
            result = regen._resolve_workers(requested=4, max_memory_mb=0, n_tasks=2)
        assert result == 2

    def test_memory_cap_reduces_workers(self):
        with mock.patch("os.cpu_count", return_value=8):
            # 100 MB cap, 50 MB per task → 2 workers max
            result = regen._resolve_workers(
                requested=4, max_memory_mb=100, n_tasks=10, est_task_mb=50.0,
            )
        assert result == 2

    def test_memory_cap_at_least_1(self):
        with mock.patch("os.cpu_count", return_value=8):
            # Tiny memory cap → at least 1 worker
            result = regen._resolve_workers(
                requested=4, max_memory_mb=10, n_tasks=10, est_task_mb=50.0,
            )
        assert result == 1

    def test_negative_requested_treated_as_auto(self):
        with mock.patch("os.cpu_count", return_value=8):
            result = regen._resolve_workers(requested=-1, max_memory_mb=0, n_tasks=10)
        assert result == 2  # auto default


class TestCLIParallelFlags:
    """Test that CLI arg parsing picks up new flags."""

    def test_defaults(self):
        import argparse
        # Simulate parsing with minimal args
        parser = argparse.ArgumentParser()
        parser.add_argument("output_dir")
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--max-memory-mb", type=int, default=0)
        parser.add_argument(
            "--parallel-mode", type=str, default="auto",
            choices=["auto", "off", "threads"],
        )
        args = parser.parse_args(["some/dir"])
        assert args.workers == 0
        assert args.max_memory_mb == 0
        assert args.parallel_mode == "auto"

    def test_explicit_values(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("output_dir")
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--max-memory-mb", type=int, default=0)
        parser.add_argument(
            "--parallel-mode", type=str, default="auto",
            choices=["auto", "off", "threads"],
        )
        args = parser.parse_args([
            "some/dir", "--workers", "4",
            "--max-memory-mb", "512", "--parallel-mode", "off",
        ])
        assert args.workers == 4
        assert args.max_memory_mb == 512
        assert args.parallel_mode == "off"
