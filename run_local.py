#!/usr/bin/env python3
"""
Nightshift Local — run everything on one machine (e.g., RunPod with Ollama).

Usage:
    python run_local.py --model granite4:small-h --num 100
"""

import argparse

from agent.local_executor import LocalExecutor
from agent.logger import ExperimentLogger
from agent.loop import NightshiftLoop
from agent.scientist import Scientist


def main():
    parser = argparse.ArgumentParser(description="Nightshift Local: All-in-one autonomous research")
    parser.add_argument("--model", default="granite4:small-h", help="Ollama model for scientist")
    parser.add_argument("--num", type=int, default=100, help="Number of experiments to run")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between experiments")
    parser.add_argument("--experiments-dir", default="experiments", help="Local experiments directory")
    args = parser.parse_args()

    scientist = Scientist(model=args.model)
    executor = LocalExecutor(scripts_dir="scripts")
    logger = ExperimentLogger(base_dir=args.experiments_dir)

    loop = NightshiftLoop(
        scientist=scientist,
        executor=executor,
        logger=logger,
    )

    loop.run(num_experiments=args.num, delay_between=args.delay)


if __name__ == "__main__":
    main()
