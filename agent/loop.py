"""
Main orchestration loop for Nightshift.
Ties together: Scientist (propose) -> Executor (train) -> Logger (record) -> repeat.
"""

import os
import tempfile
import time
from dataclasses import dataclass

import yaml
from rich.console import Console
from rich.panel import Panel

from .executor import RunPodExecutor
from .logger import ExperimentLogger
from .scientist import Scientist

console = Console()


@dataclass
class NightshiftLoop:
    """The autonomous experiment loop."""

    scientist: Scientist
    executor: RunPodExecutor
    logger: ExperimentLogger
    program_path: str = "program.md"
    config_path: str = "experiment.yaml"
    leaderboard_path: str = "leaderboard.md"

    def _load_program(self) -> str:
        with open(self.program_path) as f:
            return f.read()

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _save_config(self, config: dict) -> None:
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def run_single(self, experiment_id: str | None = None) -> dict:
        """Run a single experiment cycle: propose -> train -> evaluate -> log."""
        if experiment_id is None:
            experiment_id = self.logger.next_experiment_id()

        console.print(Panel(f"[bold]Experiment {experiment_id}", style="blue"))

        program = self._load_program()
        current_config = self._load_config()
        history = self.logger.get_history()

        if history:
            console.print("[cyan]Asking scientist for next experiment...")
            try:
                new_config, reasoning = self.scientist.propose(program, current_config, history)
                description = reasoning[:200]
            except ValueError as e:
                console.print(f"[bold red]Scientist failed: {e}")
                return {"status": "scientist_failed", "error": str(e)}
        else:
            console.print("[cyan]First run — using default config as baseline.")
            new_config = current_config
            reasoning = "Baseline run with default settings."
            description = "baseline"

        self._save_config(new_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(new_config, tmp, default_flow_style=False)
            tmp_path = tmp.name

        try:
            metrics = self.executor.run_experiment(tmp_path, experiment_id)
        except Exception as e:
            console.print(f"[bold red]Executor error: {e}")
            metrics = {"status": "crashed", "error": str(e)}
        finally:
            os.unlink(tmp_path)

        metrics["description"] = description
        self.logger.log(experiment_id, new_config, metrics, reasoning)
        self.logger.save_leaderboard(self.leaderboard_path)

        if metrics.get("status") == "completed":
            console.print(
                f"[bold green]MAE: {metrics.get('mae', 'N/A')} | "
                f"MASE: {metrics.get('mase', 'N/A')} | "
                f"Time: {metrics.get('wall_time_seconds', 'N/A')}s"
            )
        else:
            console.print(f"[bold red]Status: {metrics.get('status')} — {metrics.get('error', '')[:200]}")

        return metrics

    def run(self, num_experiments: int = 100, delay_between: int = 10) -> None:
        """Run the full autonomous loop."""
        host = getattr(self.executor, "host", "local")
        console.print(Panel(
            f"[bold]Nightshift starting — {num_experiments} experiments\n"
            f"Model: {self.scientist.model}\n"
            f"Executor: {host}",
            title="NIGHTSHIFT",
            style="bold blue",
        ))

        for i in range(num_experiments):
            experiment_id = self.logger.next_experiment_id()

            console.print(f"\n{'='*60}")
            console.print(f"[bold]Run {i+1}/{num_experiments}")
            console.print(f"{'='*60}\n")

            result = self.run_single(experiment_id)

            if i < num_experiments - 1:
                console.print(f"[dim]Waiting {delay_between}s before next experiment...[/dim]")
                time.sleep(delay_between)

        console.print("\n")
        console.print(Panel(self.logger.generate_leaderboard_md(), title="FINAL LEADERBOARD"))
        console.print("[bold green]Nightshift complete!")
