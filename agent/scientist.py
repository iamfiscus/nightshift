"""
SLM Scientist: uses a local language model (via Ollama) to reason about
experiment results and propose new configurations.
"""

import re
from dataclasses import dataclass, field

import ollama
import yaml
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are an autonomous AI research scientist running experiments to optimize
a time series foundation model (TOTO) for energy demand forecasting.

Your job: analyze past experiment results and propose the NEXT experiment config
that you believe will improve the target metric (MAE — lower is better).

RULES:
1. Change ONE thing at a time to isolate effects (unless combining known-good changes).
2. Always explain your reasoning BEFORE the config.
3. Output the complete experiment.yaml in a ```yaml``` code block.
4. The config must be valid YAML matching the schema exactly.
5. max_steps MUST equal warmup_steps + stable_steps + decay_steps.
6. Stay within parameter ranges specified in the program.

You are methodical, curious, and data-driven. You build on what works."""


@dataclass
class Scientist:
    """Local SLM agent that proposes experiment configurations."""

    model: str = "granite3.1-dense:2b"

    def _build_prompt(self, program: str, current_config: dict, history: list[dict]) -> str:
        parts = []
        parts.append("## Research Program\n")
        parts.append(program)
        parts.append("\n\n## Current Best Config\n```yaml\n")
        parts.append(yaml.dump(current_config, default_flow_style=False))
        parts.append("```\n")

        if history:
            parts.append("\n## Experiment History (most recent last)\n")
            parts.append("| ID | MAE | Key Change | Status |\n")
            parts.append("|---|---|---|---|\n")
            for exp in history[-20:]:
                mae = exp.get("mae", "N/A")
                if isinstance(mae, float):
                    mae = f"{mae:.2f}"
                change = exp.get("description", "baseline")
                status = exp.get("status", "completed")
                parts.append(f"| {exp['id']} | {mae} | {change} | {status} |\n")

            completed = [e for e in history if e.get("status") == "completed" and isinstance(e.get("mae"), (int, float))]
            if completed:
                best = min(completed, key=lambda e: e["mae"])
                parts.append(f"\n**Best so far:** {best['id']} with MAE={best['mae']:.2f}\n")

        parts.append("\n## Your Task\n")
        parts.append("Propose the next experiment. Explain your reasoning, then output the complete experiment.yaml.\n")

        return "".join(parts)

    def _parse_response(self, response_text: str) -> tuple[dict, str]:
        """Extract YAML config and reasoning from LLM response."""
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", response_text, re.DOTALL)
        if not yaml_match:
            raise ValueError("No YAML block found in response")

        config = yaml.safe_load(yaml_match.group(1))

        reasoning = response_text[:yaml_match.start()].strip()
        if not reasoning:
            reasoning = "No reasoning provided."

        return config, reasoning

    def _validate_config(self, config: dict) -> list[str]:
        """Validate the proposed config. Returns list of errors (empty = valid)."""
        errors = []

        for key in ["dataset", "target", "horizon", "seasonality", "covariates", "data", "training", "evaluation"]:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        if "training" in config:
            t = config["training"]
            expected = t.get("warmup_steps", 0) + t.get("stable_steps", 0) + t.get("decay_steps", 0)
            if t.get("max_steps") != expected:
                errors.append(f"max_steps ({t.get('max_steps')}) != warmup+stable+decay ({expected})")

            lr = t.get("learning_rate", 0)
            if not (1e-5 <= lr <= 1e-3):
                errors.append(f"learning_rate {lr} outside range [1e-5, 1e-3]")

        if "data" in config:
            d = config["data"]
            if d.get("context_factor", 0) not in [4, 8, 16]:
                errors.append(f"context_factor must be 4, 8, or 16 (got {d.get('context_factor')})")
            if d.get("train_batch_size", 0) not in [4, 8, 16, 32]:
                errors.append(f"train_batch_size must be 4, 8, 16, or 32 (got {d.get('train_batch_size')})")

        if config.get("dataset") != "proenfo_gfc12":
            errors.append("dataset must be proenfo_gfc12")
        if config.get("horizon") != 168:
            errors.append("horizon must be 168")

        return errors

    def propose(self, program: str, current_config: dict, history: list[dict], max_retries: int = 3) -> tuple[dict, str]:
        """Ask the SLM to propose the next experiment."""
        prompt = self._build_prompt(program, current_config, history)

        for attempt in range(max_retries):
            console.print(f"[cyan]Scientist thinking (attempt {attempt + 1}/{max_retries})...")

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.message.content
            console.print(f"[dim]{response_text[:200]}...[/dim]")

            try:
                config, reasoning = self._parse_response(response_text)
            except ValueError as e:
                console.print(f"[yellow]Parse error: {e}. Retrying...")
                prompt += f"\n\nYour previous response didn't contain a valid YAML block. Please try again with ```yaml ... ``` format."
                continue

            errors = self._validate_config(config)
            if errors:
                console.print(f"[yellow]Validation errors: {errors}. Retrying...")
                prompt += f"\n\nYour previous config had errors: {errors}. Please fix and try again."
                continue

            return config, reasoning

        raise ValueError(f"Scientist failed to produce valid config after {max_retries} attempts")
