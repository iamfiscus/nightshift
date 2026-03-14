"""Tests for the SLM scientist agent."""
import json
import pytest
from unittest.mock import patch, MagicMock


def test_build_prompt_includes_program():
    """Scientist prompt includes the research program."""
    from agent.scientist import Scientist

    scientist = Scientist(model="granite3.1-dense:2b")
    prompt = scientist._build_prompt(
        program="Minimize MAE on energy data",
        current_config={"training": {"learning_rate": 0.00004}},
        history=[],
    )
    assert "Minimize MAE" in prompt
    assert "learning_rate" in prompt


def test_build_prompt_includes_history():
    """Scientist prompt includes past experiment results."""
    from agent.scientist import Scientist

    scientist = Scientist(model="granite3.1-dense:2b")
    history = [
        {"id": "exp_001", "config": {"training": {"learning_rate": 0.00004}}, "mae": 150.0},
        {"id": "exp_002", "config": {"training": {"learning_rate": 0.0001}}, "mae": 140.0},
    ]
    prompt = scientist._build_prompt(
        program="Minimize MAE",
        current_config={"training": {"learning_rate": 0.0001}},
        history=history,
    )
    assert "exp_001" in prompt
    assert "150.0" in prompt
    assert "140.0" in prompt


def test_parse_yaml_from_response():
    """Scientist extracts YAML from LLM response."""
    from agent.scientist import Scientist

    scientist = Scientist(model="granite3.1-dense:2b")
    response = """I think we should increase the learning rate.

```yaml
dataset: proenfo_gfc12
target: target
horizon: 168
seasonality: 24
covariates:
  include: [airtemperature]
data:
  context_factor: 8
  num_train_samples: 100
  max_rows: 5000
  train_batch_size: 16
  val_batch_size: 1
training:
  learning_rate: 0.0001
  warmup_steps: 1000
  stable_steps: 200
  decay_steps: 200
  max_steps: 1400
evaluation:
  metric: mae
```

This should improve results because higher LR converges faster."""

    config, reasoning = scientist._parse_response(response)
    assert config["training"]["learning_rate"] == 0.0001
    assert config["covariates"]["include"] == ["airtemperature"]
    assert "increase the learning rate" in reasoning


def test_validate_config_catches_bad_max_steps():
    """Validator catches when max_steps != warmup + stable + decay."""
    from agent.scientist import Scientist

    scientist = Scientist(model="granite3.1-dense:2b")
    bad_config = {
        "dataset": "proenfo_gfc12",
        "target": "target",
        "horizon": 168,
        "seasonality": 24,
        "covariates": {"include": []},
        "data": {"context_factor": 8, "num_train_samples": 100, "max_rows": 5000, "train_batch_size": 16, "val_batch_size": 1},
        "training": {"learning_rate": 0.0001, "warmup_steps": 1000, "stable_steps": 200, "decay_steps": 200, "max_steps": 9999},
        "evaluation": {"metric": "mae"},
    }
    errors = scientist._validate_config(bad_config)
    assert any("max_steps" in e for e in errors)


def test_validate_config_accepts_valid():
    """Validator accepts a valid config."""
    from agent.scientist import Scientist

    scientist = Scientist(model="granite3.1-dense:2b")
    good_config = {
        "dataset": "proenfo_gfc12",
        "target": "target",
        "horizon": 168,
        "seasonality": 24,
        "covariates": {"include": []},
        "data": {"context_factor": 8, "num_train_samples": 100, "max_rows": 5000, "train_batch_size": 16, "val_batch_size": 1},
        "training": {"learning_rate": 0.0001, "warmup_steps": 1000, "stable_steps": 200, "decay_steps": 200, "max_steps": 1400},
        "evaluation": {"metric": "mae"},
    }
    errors = scientist._validate_config(good_config)
    assert errors == []
