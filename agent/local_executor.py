"""
Local executor: runs TOTO fine-tuning on the same machine (no SSH).
Used when scientist + training run on the same RunPod instance.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class LocalExecutor:
    """Executes TOTO fine-tuning locally via subprocess."""

    scripts_dir: str = "scripts"

    def run_experiment(self, experiment_yaml: str, experiment_id: str) -> dict:
        """
        Run training locally via subprocess.

        Args:
            experiment_yaml: Path to the experiment config YAML
            experiment_id: Unique experiment identifier

        Returns:
            Dict with metrics (mae, mase, wql, train_time_seconds, etc.)
        """
        output_file = f"metrics_{experiment_id}.json"
        train_script = os.path.join(self.scripts_dir, "train_remote.py")
        finetune_config = os.path.join(self.scripts_dir, "finetune_config.yaml")

        # Copy finetune_config.yaml to cwd if not already there
        # (train_remote.py expects it in cwd)
        if not os.path.exists("finetune_config.yaml"):
            import shutil
            shutil.copy(finetune_config, "finetune_config.yaml")

        cmd = [
            "python", train_script,
            "--config", experiment_yaml,
            "--output", output_file,
        ]

        console.print(f"[yellow]Running experiment {experiment_id} locally...")
        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                console.print(f"[bold red]Experiment {experiment_id} FAILED (exit {result.returncode})")
                console.print(f"[red]{result.stderr[-500:]}")
                return {
                    "status": "crashed",
                    "error": result.stderr[-500:],
                    "wall_time_seconds": round(elapsed, 1),
                }

            # Read metrics
            if os.path.exists(output_file):
                with open(output_file) as f:
                    metrics = json.load(f)
                metrics["status"] = "completed"
                metrics["wall_time_seconds"] = round(elapsed, 1)
                # Clean up
                os.unlink(output_file)
            else:
                metrics = {
                    "status": "no_metrics",
                    "error": "metrics file not created",
                    "wall_time_seconds": round(elapsed, 1),
                }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            console.print(f"[bold red]Experiment {experiment_id} TIMED OUT after 30min")
            metrics = {
                "status": "crashed",
                "error": "timeout after 1800s",
                "wall_time_seconds": round(elapsed, 1),
            }

        console.print(
            f"[bold green]Experiment {experiment_id} completed in {elapsed:.0f}s "
            f"— MAE: {metrics.get('mae', 'N/A')}"
        )
        return metrics
