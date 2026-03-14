#!/usr/bin/env python3
"""
Nightshift remote training script.
Runs on RunPod GPU. Reads experiment config, fine-tunes TOTO, writes metrics.

Usage: python train_remote.py --config experiment.yaml --output metrics.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Nightshift TOTO fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    parser.add_argument("--output", type=str, default="metrics.json", help="Output metrics file")
    args = parser.parse_args()

    with open(args.config) as f:
        exp_config = yaml.safe_load(f)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    torch.use_deterministic_algorithms(True)

    from toto.scripts.benchmark_finetuning import (
        load_config,
        prepare_dataset,
        get_config,
        aggregate_results,
    )
    from toto.scripts import finetune_toto as finetune
    from toto.evaluation.fev.evaluate import evaluate_model

    start_time = time.time()

    base_config = load_config("finetune_config.yaml")

    base_config["model"]["lr"] = exp_config["training"]["learning_rate"]
    base_config["model"]["warmup_steps"] = exp_config["training"]["warmup_steps"]
    base_config["model"]["stable_steps"] = exp_config["training"]["stable_steps"]
    base_config["model"]["decay_steps"] = exp_config["training"]["decay_steps"]
    base_config["trainer"]["max_steps"] = exp_config["training"]["max_steps"]
    base_config["data"]["train_batch_size"] = exp_config["data"]["train_batch_size"]
    base_config["data"]["context_factor"] = exp_config["data"]["context_factor"]
    base_config["data"]["num_train_samples"] = exp_config["data"]["num_train_samples"]
    base_config["data"]["max_rows"] = exp_config["data"]["max_rows"]

    use_ev = len(exp_config.get("covariates", {}).get("include", [])) > 0
    ev_fields = exp_config["covariates"]["include"] if use_ev else []

    dataset_name = exp_config["dataset"]
    target_fields = ["target"]
    custom_dataset = prepare_dataset(dataset_name, target_fields, ev_fields)

    config = get_config(
        base_config=base_config,
        model_name="nightshift_run",
        dataset_name=dataset_name,
        add_exogenous_features=use_ev,
        horizon=exp_config["horizon"],
    )

    lightning_module, patch_size = finetune.init_lightning(config)
    datamodule = finetune.get_datamodule(config, patch_size, custom_dataset, setup=True)

    trained_module, best_ckpt_path, best_val_loss = finetune.train(
        lightning_module, datamodule, config
    )

    train_time = time.time() - start_time

    eval_start = time.time()
    if best_ckpt_path:
        model = finetune.load_finetuned_toto(
            config["pretrained_model"],
            best_ckpt_path,
            lightning_module.device,
        )
    else:
        model = trained_module

    results = evaluate_model(
        model,
        datamodule._view.hf_dataset,
        context_length=datamodule._view._context_length,
        prediction_length=exp_config["horizon"],
        seasonality=exp_config["seasonality"],
        stride=exp_config["horizon"],
        add_exogenous_variables=use_ev,
        num_samples=64,
        samples_per_batch=64,
    )
    eval_time = time.time() - eval_start

    agg = aggregate_results(results)
    metrics = {
        "mae": float(agg.get("abs_error", float("inf"))),
        "mase": float(agg.get("MASE", float("inf"))),
        "wql": float(agg.get("mean_wQuantileLoss", float("inf"))),
        "val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "train_time_seconds": round(train_time, 1),
        "eval_time_seconds": round(eval_time, 1),
        "best_checkpoint": best_ckpt_path,
        "config": exp_config,
    }

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== NIGHTSHIFT RESULTS ===")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MASE: {metrics['mase']:.4f}")
    print(f"WQL: {metrics['wql']:.4f}")
    print(f"Val Loss: {metrics['val_loss']}")
    print(f"Train time: {metrics['train_time_seconds']}s")
    print(f"Eval time: {metrics['eval_time_seconds']}s")
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
