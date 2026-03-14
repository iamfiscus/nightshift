"""
RunPod executor: uploads experiment config, runs TOTO fine-tuning via SSH,
downloads metrics. Handles the local-remote bridge.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko
from rich.console import Console

console = Console()


@dataclass
class RunPodExecutor:
    """Executes TOTO fine-tuning on a remote RunPod instance via SSH."""

    host: str
    user: str = "root"
    key_path: str = ""
    port: int = 22
    remote_dir: str = "/workspace/nightshift"
    password: str = ""

    def _get_ssh_client(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": self.host,
            "username": self.user,
            "port": self.port,
        }
        if self.key_path:
            connect_kwargs["key_filename"] = self.key_path
        elif self.password:
            connect_kwargs["password"] = self.password
        client.connect(**connect_kwargs)
        return client

    def _build_run_command(self, config_name: str, output_name: str) -> str:
        return (
            f"cd {self.remote_dir} && "
            f"python train_remote.py --config {config_name} --output {output_name}"
        )

    def _parse_metrics(self, raw_json: str) -> dict:
        return json.loads(raw_json)

    def setup_remote(self, local_scripts_dir: str) -> None:
        """One-time setup: upload training scripts to RunPod."""
        console.print(f"[bold blue]Setting up remote at {self.host}...")
        client = self._get_ssh_client()
        sftp = client.open_sftp()

        try:
            sftp.mkdir(self.remote_dir)
        except IOError:
            pass

        scripts = ["train_remote.py", "finetune_config.yaml"]
        for script in scripts:
            local_path = os.path.join(local_scripts_dir, script)
            remote_path = f"{self.remote_dir}/{script}"
            if os.path.exists(local_path):
                sftp.put(local_path, remote_path)
                console.print(f"  Uploaded {script}")

        sftp.close()
        client.close()
        console.print("[bold green]Remote setup complete.")

    def run_experiment(self, experiment_yaml: str, experiment_id: str) -> dict:
        """Upload experiment config, run training, download metrics."""
        config_name = f"experiment_{experiment_id}.yaml"
        output_name = f"metrics_{experiment_id}.json"

        client = self._get_ssh_client()
        sftp = client.open_sftp()

        remote_config = f"{self.remote_dir}/{config_name}"
        sftp.put(experiment_yaml, remote_config)
        console.print(f"[blue]Uploaded config for {experiment_id}")

        cmd = self._build_run_command(config_name, output_name)
        console.print(f"[yellow]Running experiment {experiment_id}...")
        start = time.time()

        stdin, stdout, stderr = client.exec_command(cmd, timeout=1800)
        exit_code = stdout.channel.recv_exit_status()
        elapsed = time.time() - start

        if exit_code != 0:
            err = stderr.read().decode()
            console.print(f"[bold red]Experiment {experiment_id} FAILED (exit {exit_code})")
            console.print(f"[red]{err[-500:]}")
            sftp.close()
            client.close()
            return {
                "status": "crashed",
                "error": err[-500:],
                "wall_time_seconds": round(elapsed, 1),
            }

        remote_output = f"{self.remote_dir}/{output_name}"
        try:
            with sftp.open(remote_output, "r") as f:
                metrics = self._parse_metrics(f.read().decode())
            metrics["status"] = "completed"
            metrics["wall_time_seconds"] = round(elapsed, 1)
        except Exception as e:
            metrics = {
                "status": "no_metrics",
                "error": str(e),
                "wall_time_seconds": round(elapsed, 1),
            }

        sftp.close()
        client.close()

        console.print(
            f"[bold green]Experiment {experiment_id} completed in {elapsed:.0f}s "
            f"— MAE: {metrics.get('mae', 'N/A')}"
        )
        return metrics
