"""
=============================================================================
FUMD-AI Training Workflow -- shared library: minimal notebook step-runner
=============================================================================
Used by: run_pipeline.ipynb (local/interactive convenience orchestration)

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

This project's actual execution environment invokes one notebook at a time
via `jupyter execute <notebook>.ipynb` (nbclient's own CLI) inside a Slurm
job's Singularity/Apptainer container -- see slurm/*.sbatch -- with no
parameter-injection mechanism of its own. Deliberately built directly on
`nbformat`/`nbclient` (both already required by `nbconvert`, which this
project's execution image includes) instead of adding a `papermill`
dependency that may not be installable inside a managed HPC image.

`run_step` does the same core trick papermill is built around: overwrite
the source of the notebook's single cell tagged `"parameters"` before
running it, so the same notebook file works both as a manually-edited,
standalone `jupyter execute`/Slurm submission (see each step notebook's own
parameters cell) and as one stage of `run_pipeline.ipynb`'s local/
interactive chain.
"""

from __future__ import annotations

import json
import os


def inject_parameters(nb, parameters: dict):
    """Append `name = <python-literal>` assignments to the end of the cell
    tagged `"parameters"`, one per key in `parameters`. Raises if no such
    cell exists -- every step notebook in this workflow has exactly one.

    Appends rather than replaces the cell's source: a caller (e.g.
    run_pipeline.ipynb) is only expected to override the subset of
    parameters it actually cares about propagating between steps -- every
    *other* parameter must keep resolving to the value the notebook's own
    parameters cell already assigns it. Replacing the whole cell source
    with only the given keys would silently undefine every parameter left
    out of `parameters`, which is exactly the bug this used to have: e.g.
    run_pipeline.ipynb's call for Step 2 only passes INPUT_DIR, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE, and RANDOM_SEED --
    LSTM_UNITS/ATTENTION_UNITS/EMBEDDING_DIM/DENSE_UNITS/DROPOUT_RATE/
    LEARNING_RATE/VALIDATION_SPLIT are meant to keep the notebook's own
    defaults. Since cell execution is top-to-bottom, appending re-assigns
    only the given names, in place, after the originals have already run.

    Uses `repr()`, not `json.dumps()`, to render each value: JSON and
    Python literal syntax aren't the same thing -- `json.dumps(None)` is
    the string "null", which is valid JSON but not valid Python (and
    likewise "true"/"false" vs. Python's True/False), so writing it
    directly into a code cell as `NAME = null` is a NameError waiting to
    happen. `repr()` on the plain str/int/float/bool/None/list/dict values
    these parameters are always built from round-trips correctly as
    executable Python source.
    """
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.get("cell_type") == "code" and "parameters" in tags:
            lines = [cell["source"], "", "# --- overridden by notebook_runner.run_step ---"]
            lines += [f"{name} = {value!r}" for name, value in parameters.items()]
            cell["source"] = "\n".join(lines)
            return nb
    raise ValueError("no cell tagged 'parameters' found in this notebook")


def run_step(
    notebook_path: str,
    output_path: str,
    parameters: dict | None = None,
    *,
    cwd: str | None = None,
    kernel_name: str = "python3",
    timeout: int | None = None,
):
    """
    Load `notebook_path`, optionally inject `parameters` into its
    `"parameters"`-tagged cell, execute it top to bottom (via nbclient --
    the same mechanism `jupyter execute` uses), and write the executed
    notebook (with outputs) to `output_path`.

    `cwd` should normally be the repository root (not `notebooks/`) since
    every step notebook resolves paths like `src/`, `example-data/`, and
    its `OUTPUT_DIR` relative to the repository root -- matching how the
    preprocessing workflow's notebooks are documented to be run.
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(notebook_path, as_version=4)
    if parameters:
        inject_parameters(nb, parameters)

    # Set the kernel's working directory the same way the `jupyter execute`
    # CLI itself does (resources['metadata']['path']), rather than passing
    # cwd= to NotebookClient.execute() -- that kwarg is not a documented
    # NotebookClient.execute() parameter (it flows through **kwargs to
    # kernel startup, which is not a reliable place to depend on), while
    # resources['metadata']['path'] is nbclient's own supported mechanism.
    resources = {"metadata": {"path": cwd or os.getcwd()}}
    client = NotebookClient(nb, kernel_name=kernel_name, timeout=timeout, resources=resources)
    client.execute()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nbformat.write(nb, output_path)
    return nb


def env_override(name: str, default):
    """Return `os.environ[name]` if set, else `default` unchanged.

    `jupyter execute` (unlike papermill) has no `-p NAME VALUE` flag -- the
    only way to change a parameter without hand-editing the notebook file
    is an environment variable read at execution time, e.g.:

        DATASET_PATH=/abs/path/dataset_labeled_w3.csv \
        OUTPUT_DIR=/abs/path/pipeline_run \
        jupyter execute notebooks/step_1_load_and_window_dataset.ipynb

    The environment variable's value is JSON-decoded first (so ints,
    floats, lists, dicts, null, and booleans all round-trip correctly --
    e.g. `FUTURE_STEPS='[1,2,3]'` or `SEQUENCE_LENGTH=6`); if it isn't
    valid JSON, it's used as-is (the common case: a plain path string like
    `/abs/path/dataset_labeled_w3.csv`, which is not valid JSON on its
    own). Under Slurm/Singularity, `singularity exec` (without
    `--cleanenv`, which this project's sbatch templates do not use) passes
    the submitting shell's environment through into the container, so
    `export`-ing these before `sbatch`/`singularity exec` works the same
    way. Each step notebook's own parameters-cell default is used when the
    variable is unset.
    """
    if name not in os.environ:
        return default
    raw = os.environ[name]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
