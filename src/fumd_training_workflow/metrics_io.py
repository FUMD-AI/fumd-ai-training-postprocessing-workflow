"""
=============================================================================
FUMD-AI Training Workflow -- shared library: structured metrics I/O
=============================================================================
Used by: notebooks/step_2_train_model.ipynb, step_3_evaluate_model.ipynb,
step_4_aggregate_training_metrics.ipynb

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

The original exploratory notebooks printed everything to stdout, redirected
into a per-run `.txt` file (via a manually-opened `f = open(...)` used
throughout the training notebook), and then `leer_metricas.ipynb` read that
free text back with regular expressions to build comparison tables. That's
brittle -- any change in wording/spacing breaks the regex silently.

This module is the replacement: Step 3 writes each evaluation run's metrics
here directly as CSV + JSON; Step 4 (and anyone else) reads them back
without parsing prose.
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd


def save_run_manifest(output_dir: str, manifest: dict) -> str:
    """Persist a training run's configuration (Step 2) -- feature list,
    sequence_length, future_steps, num_base_stations, source dataset, model
    hyperparameters -- so later steps can reload it without guessing."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def load_run_manifest(output_dir: str) -> dict:
    with open(os.path.join(output_dir, "run_manifest.json")) as f:
        return json.load(f)


def write_metrics(
    output_dir: str,
    run_label: str,
    per_output: list[dict],
    overall: dict,
    migration_breakdown: dict | None = None,
    meta: dict | None = None,
) -> tuple[str, str]:
    """
    Write one evaluation run's metrics (Step 3's output for one dataset) as
    both a flat CSV (per forecast step -- easy to `pd.concat` across runs
    for Step 4's comparison plots) and a full JSON (keeps the confusion
    matrices / per-class breakdown the CSV can't hold in a flat table).

    `run_label` identifies *what was evaluated* (e.g. "train_val", or the
    name of a cross-run generalization dataset like "pipeline_run_1000_2")
    -- not the model; `meta` is the place for model/run identity if you're
    comparing multiple trained models (e.g. {"model_run": "...", ...}).
    """
    os.makedirs(output_dir, exist_ok=True)

    flat_rows = []
    for row in per_output:
        flat = {k: v for k, v in row.items() if k not in ("confusion_matrix", "per_class")}
        flat["run_label"] = run_label
        flat_rows.append(flat)
    csv_path = os.path.join(output_dir, f"metrics_{run_label}.csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    payload = {
        "run_label": run_label,
        "overall": overall,
        "per_output": per_output,
        "migration_breakdown": migration_breakdown,
        "meta": meta or {},
    }
    json_path = os.path.join(output_dir, f"metrics_{run_label}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path


def read_metrics_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def discover_metrics(root_dir: str, pattern: str = "metrics_*.json") -> list[dict]:
    """Find and load every `metrics_<run_label>.json` under `root_dir`
    (recursively) -- e.g. across several Step 3 runs / model variants, for
    Step 4's comparison tables and plots."""
    paths = sorted(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))
    return [read_metrics_json(p) for p in paths]


def to_long_dataframe(metrics_list: list[dict], variant_label: str | None = None) -> pd.DataFrame:
    """
    Flatten a list of `write_metrics`-style payloads (as loaded by
    `discover_metrics`) into one long-format DataFrame:
    columns = [variant, run_label, future_step, metric, value] -- ready for
    a `seaborn.barplot`/`catplot` comparison across datasets/outputs, the
    same shape `leer_metricas.ipynb` built by hand from regex matches.

    `variant_label` is an optional extra column (e.g. a model/architecture
    name) for comparing more than one trained model side by side.
    """
    metric_cols = ["precision", "recall", "f1", "top2_accuracy", "accuracy"]
    rows = []
    for payload in metrics_list:
        run_label = payload["run_label"]
        for step_row in payload["per_output"]:
            for metric in metric_cols:
                rows.append({
                    "variant": variant_label,
                    "run_label": run_label,
                    "future_step": step_row["future_step"],
                    "metric": metric,
                    "value": step_row[metric],
                })
    return pd.DataFrame(rows)
