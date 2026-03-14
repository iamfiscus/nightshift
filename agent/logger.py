"""
Experiment logger: stores configs, metrics, and reasoning per experiment.
Generates a ranked leaderboard.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ExperimentLogger:
    """Logs experiments and generates leaderboards."""

    base_dir: str = "experiments"

    def log(self, experiment_id: str, config: dict, metrics: dict, reasoning: str) -> None:
        """Log a single experiment."""
        exp_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)

        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        metrics["experiment_id"] = experiment_id
        metrics["timestamp"] = datetime.now().isoformat()
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(exp_dir, "reasoning.md"), "w") as f:
            f.write(f"# Experiment {experiment_id}\n\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Reasoning\n\n{reasoning}\n")

    def _load_experiment(self, experiment_id: str) -> dict:
        exp_dir = os.path.join(self.base_dir, experiment_id)
        metrics_path = os.path.join(exp_dir, "metrics.json")
        config_path = os.path.join(exp_dir, "config.yaml")
        reasoning_path = os.path.join(exp_dir, "reasoning.md")

        result = {"id": experiment_id}

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                result.update(json.load(f))

        if os.path.exists(config_path):
            with open(config_path) as f:
                result["config"] = yaml.safe_load(f)

        if os.path.exists(reasoning_path):
            with open(reasoning_path) as f:
                result["reasoning_text"] = f.read()

        return result

    def get_history(self) -> list[dict]:
        """Get all experiments in chronological order."""
        if not os.path.exists(self.base_dir):
            return []

        experiments = []
        for name in sorted(os.listdir(self.base_dir)):
            exp_dir = os.path.join(self.base_dir, name)
            if os.path.isdir(exp_dir) and os.path.exists(os.path.join(exp_dir, "metrics.json")):
                experiments.append(self._load_experiment(name))

        return experiments

    def get_leaderboard(self) -> list[dict]:
        """Get completed experiments sorted by MAE (ascending = best first)."""
        history = self.get_history()
        completed = [
            e for e in history
            if e.get("status") == "completed" and isinstance(e.get("mae"), (int, float))
        ]
        return sorted(completed, key=lambda e: e["mae"])

    def generate_leaderboard_md(self) -> str:
        """Generate a markdown leaderboard."""
        board = self.get_leaderboard()
        lines = []
        lines.append("# Nightshift Leaderboard\n")
        lines.append(f"_Generated: {datetime.now().isoformat()}_\n")
        lines.append(f"_Dataset: proenfo_gfc12 (hourly electricity load)_\n\n")
        lines.append("| Rank | ID | MAE | MASE | WQL | LR | Covariates | Train Time |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")

        for i, exp in enumerate(board, 1):
            config = exp.get("config", {})
            lr = config.get("training", {}).get("learning_rate", "?")
            covs = config.get("covariates", {}).get("include", [])
            cov_str = ", ".join(covs) if covs else "none"
            train_time = exp.get("train_time_seconds", "?")
            if isinstance(train_time, (int, float)):
                train_time = f"{train_time:.0f}s"

            mae_val = exp.get('mae', 0)
            mase_val = exp.get('mase', 'N/A')
            wql_val = exp.get('wql', 'N/A')

            mae_str = f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else str(mae_val)
            mase_str = f"{mase_val:.4f}" if isinstance(mase_val, (int, float)) else str(mase_val)
            wql_str = f"{wql_val:.4f}" if isinstance(wql_val, (int, float)) else str(wql_val)

            lines.append(
                f"| {i} | {exp['id']} | {mae_str} | "
                f"{mase_str} | {wql_str} | "
                f"{lr} | {cov_str} | {train_time} |\n"
            )

        history = self.get_history()
        crashes = [e for e in history if e.get("status") == "crashed"]
        if crashes:
            lines.append(f"\n## Crashed ({len(crashes)})\n")
            for exp in crashes:
                lines.append(f"- **{exp['id']}**: {exp.get('error', 'unknown error')[:100]}\n")

        return "".join(lines)

    def save_leaderboard(self, output_path: str = "leaderboard.md") -> None:
        """Write leaderboard markdown to file."""
        md = self.generate_leaderboard_md()
        with open(output_path, "w") as f:
            f.write(md)

    def next_experiment_id(self) -> str:
        """Generate the next sequential experiment ID."""
        history = self.get_history()
        return f"exp_{len(history) + 1:03d}"
