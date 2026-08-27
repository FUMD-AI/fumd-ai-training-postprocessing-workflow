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
    """Overwrite the source of the cell tagged `"parameters"` with
    `name = <json-literal>` assignments. Raises if no such cell exists --
    every step notebook in this workflow has exactly one."""
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.get("cell_type") == "code" and "parameters" in tags:
            lines = ["# --- overwritten by notebook_runner.run_step ---"]
            lines += [f"{name} = {json.dumps(value)}" for name, value in parameters.items()]
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

    client = NotebookClient(nb, kernel_name=kernel_name, timeout=timeout)
    client.execute(cwd=cwd or os.getcwd())

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nbformat.write(nb, output_path)
    return nb
