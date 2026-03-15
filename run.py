#!/usr/bin/env python3
"""
Nightshift CLI — run the autonomous experiment loop.

Usage:
    python run.py --host <runpod-ip> --port <ssh-port> [--password <pwd>] [--num 100] [--model granite3.1-dense:2b]
    python run.py --host <runpod-ip> --port <ssh-port> --setup   # One-time remote setup
"""

import argparse
import os

from dotenv import load_dotenv
load_dotenv()

from agent.executor import RunPodExecutor
from agent.logger import ExperimentLogger
from agent.loop import NightshiftLoop
from agent.scientist import Scientist


def main():
    parser = argparse.ArgumentParser(description="Nightshift: Autonomous TOTO Research Agent")
    parser.add_argument("--host", required=True, help="RunPod SSH host")
    parser.add_argument("--port", type=int, default=22, help="RunPod SSH port")
    parser.add_argument("--user", default="root", help="SSH user")
    parser.add_argument("--key", default="", help="SSH key path")
    parser.add_argument("--password", default="", help="SSH password")
    parser.add_argument("--remote-dir", default="/workspace/nightshift", help="Remote working directory")
    parser.add_argument("--model", default="granite3.1-dense:2b", help="Ollama model for scientist")
    parser.add_argument("--num", type=int, default=100, help="Number of experiments to run")
    parser.add_argument("--delay", type=int, default=10, help="Seconds between experiments")
    parser.add_argument("--setup", action="store_true", help="One-time remote setup only")
    parser.add_argument("--experiments-dir", default="experiments", help="Local experiments directory")
    args = parser.parse_args()

    executor = RunPodExecutor(
        host=args.host,
        user=args.user,
        key_path=args.key,
        port=args.port,
        remote_dir=args.remote_dir,
        password=args.password,
    )

    if args.setup:
        scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
        executor.setup_remote(scripts_dir)
        return

    scientist = Scientist(model=args.model)
    logger = ExperimentLogger(base_dir=args.experiments_dir)

    loop = NightshiftLoop(
        scientist=scientist,
        executor=executor,
        logger=logger,
    )

    loop.run(num_experiments=args.num, delay_between=args.delay)


if __name__ == "__main__":
    main()
