"""Tests for RunPod executor — uses mocked SSH for unit tests."""
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path


def test_build_remote_command():
    """Executor builds the correct remote command string."""
    from agent.executor import RunPodExecutor

    executor = RunPodExecutor(
        host="1.2.3.4",
        user="root",
        key_path="/path/to/key",
        remote_dir="/workspace/nightshift",
    )
    cmd = executor._build_run_command("experiment.yaml", "metrics.json")
    assert "python train_remote.py" in cmd
    assert "--config experiment.yaml" in cmd
    assert "--output metrics.json" in cmd


def test_parse_metrics_json():
    """Executor correctly parses metrics.json content."""
    from agent.executor import RunPodExecutor

    executor = RunPodExecutor(
        host="1.2.3.4",
        user="root",
        key_path="/path/to/key",
        remote_dir="/workspace/nightshift",
    )
    raw = json.dumps({"mae": 123.45, "mase": 1.2, "wql": 0.05, "train_time_seconds": 300.0})
    metrics = executor._parse_metrics(raw)
    assert metrics["mae"] == 123.45
    assert metrics["train_time_seconds"] == 300.0
