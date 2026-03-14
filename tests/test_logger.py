"""Tests for experiment logger and leaderboard."""
import json
import os
import tempfile
import pytest


def test_log_experiment_creates_directory():
    """Logger creates experiment directory with config and metrics."""
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=tmpdir)
        logger.log(
            experiment_id="exp_001",
            config={"training": {"learning_rate": 0.00004}},
            metrics={"mae": 150.0, "status": "completed"},
            reasoning="Baseline run with default settings",
        )

        exp_dir = os.path.join(tmpdir, "exp_001")
        assert os.path.exists(exp_dir)
        assert os.path.exists(os.path.join(exp_dir, "config.yaml"))
        assert os.path.exists(os.path.join(exp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(exp_dir, "reasoning.md"))


def test_leaderboard_sorted_by_mae():
    """Leaderboard returns experiments sorted by MAE (ascending)."""
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=tmpdir)
        logger.log("exp_001", {}, {"mae": 150.0, "status": "completed"}, "first")
        logger.log("exp_002", {}, {"mae": 120.0, "status": "completed"}, "second")
        logger.log("exp_003", {}, {"mae": 140.0, "status": "completed"}, "third")

        board = logger.get_leaderboard()
        assert len(board) == 3
        assert board[0]["id"] == "exp_002"
        assert board[1]["id"] == "exp_003"
        assert board[2]["id"] == "exp_001"


def test_leaderboard_excludes_crashed():
    """Leaderboard excludes crashed experiments from ranking."""
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=tmpdir)
        logger.log("exp_001", {}, {"mae": 150.0, "status": "completed"}, "ok")
        logger.log("exp_002", {}, {"status": "crashed", "error": "OOM"}, "crashed")

        board = logger.get_leaderboard()
        assert len(board) == 1
        assert board[0]["id"] == "exp_001"


def test_get_history_returns_all():
    """History returns all experiments including crashes."""
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=tmpdir)
        logger.log("exp_001", {}, {"mae": 150.0, "status": "completed"}, "ok")
        logger.log("exp_002", {}, {"status": "crashed"}, "crashed")

        history = logger.get_history()
        assert len(history) == 2


def test_generate_leaderboard_markdown():
    """Logger generates a markdown leaderboard file."""
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=tmpdir)
        logger.log("exp_001", {"training": {"learning_rate": 0.00004}}, {"mae": 150.0, "status": "completed"}, "baseline")
        logger.log("exp_002", {"training": {"learning_rate": 0.0001}}, {"mae": 120.0, "status": "completed"}, "higher LR")

        md = logger.generate_leaderboard_md()
        assert "exp_002" in md
        assert "120.0" in md


def test_next_experiment_id():
    """Logger generates sequential experiment IDs."""
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(base_dir=tmpdir)
        assert logger.next_experiment_id() == "exp_001"
        logger.log("exp_001", {}, {"mae": 150.0, "status": "completed"}, "first")
        assert logger.next_experiment_id() == "exp_002"
