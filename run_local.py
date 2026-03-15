#!/usr/bin/env python3
"""
Nightshift Local — run everything on one machine (e.g., RunPod with Ollama).

Usage:
    # Ollama (local)
    python run_local.py --model granite4:small-h --num 100

    # OpenRouter (cloud)
    python run_local.py --backend openrouter --model openrouter/hunter-alpha --num 100
"""

import argparse
import os

from dotenv import load_dotenv
load_dotenv()

from agent.local_executor import LocalExecutor
from agent.logger import ExperimentLogger
from agent.loop import NightshiftLoop
from agent.scientist import Scientist


def main():
    parser = argparse.ArgumentParser(description="Nightshift Local: All-in-one autonomous research")
    parser.add_argument("--model", default="granite4:small-h", help="Model name (Ollama or OpenRouter)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openrouter"], help="LLM backend")
    parser.add_argument("--api-key", default="", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--num", type=int, default=100, help="Number of experiments to run")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between experiments")
    parser.add_argument("--experiments-dir", default="experiments", help="Local experiments directory")
    args = parser.parse_args()

    scientist = Scientist(
        model=args.model,
        backend=args.backend,
        api_key=args.api_key,
    )
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
