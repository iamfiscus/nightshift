"""End-to-end dry run: scientist + mocked executor + real logger."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

import yaml


def test_full_loop_dry_run():
    """Run 3 iterations with mocked executor and mocked Ollama."""
    from agent.loop import NightshiftLoop
    from agent.scientist import Scientist
    from agent.executor import RunPodExecutor
    from agent.logger import ExperimentLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write program.md
        program_path = os.path.join(tmpdir, "program.md")
        with open(program_path, "w") as f:
            f.write("Minimize MAE on proenfo_gfc12 energy dataset.\n")

        # Write experiment.yaml
        config_path = os.path.join(tmpdir, "experiment.yaml")
        config = {
            "dataset": "proenfo_gfc12",
            "target": "target",
            "horizon": 168,
            "seasonality": 24,
            "covariates": {"include": []},
            "data": {
                "context_factor": 8,
                "num_train_samples": 100,
                "max_rows": 5000,
                "train_batch_size": 16,
                "val_batch_size": 1,
            },
            "training": {
                "learning_rate": 0.00004,
                "warmup_steps": 1000,
                "stable_steps": 200,
                "decay_steps": 200,
                "max_steps": 1400,
            },
            "evaluation": {"metric": "mae"},
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Mock Ollama to return a valid YAML response
        yaml_response = yaml.dump(config)
        mock_ollama_response = MagicMock()
        mock_ollama_response.message.content = (
            f"Let's try the baseline first.\n\n```yaml\n{yaml_response}```\n"
        )

        # Mock executor to return fake metrics
        mock_executor = MagicMock(spec=RunPodExecutor)
        mock_executor.host = "mock-host"
        call_count = [0]
        def fake_run(config_path, exp_id):
            call_count[0] += 1
            return {
                "mae": 150.0 - call_count[0] * 10,
                "mase": 1.2,
                "wql": 0.05,
                "status": "completed",
                "train_time_seconds": 300.0,
                "wall_time_seconds": 310.0,
            }
        mock_executor.run_experiment.side_effect = fake_run

        experiments_dir = os.path.join(tmpdir, "experiments")
        leaderboard_path = os.path.join(tmpdir, "leaderboard.md")

        scientist = Scientist(model="granite3.1-dense:2b")
        logger = ExperimentLogger(base_dir=experiments_dir)

        loop = NightshiftLoop(
            scientist=scientist,
            executor=mock_executor,
            logger=logger,
            program_path=program_path,
            config_path=config_path,
            leaderboard_path=leaderboard_path,
        )

        # Run 3 experiments with mocked Ollama
        with patch("ollama.chat", return_value=mock_ollama_response):
            loop.run(num_experiments=3, delay_between=0)

        # Verify results
        history = logger.get_history()
        assert len(history) == 3

        board = logger.get_leaderboard()
        assert len(board) == 3
        assert board[0]["mae"] < board[-1]["mae"]  # Best first

        # Leaderboard file was written
        assert os.path.exists(leaderboard_path)
        with open(leaderboard_path) as f:
            md = f.read()
        assert "exp_001" in md
