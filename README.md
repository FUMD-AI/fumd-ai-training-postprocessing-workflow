# FUMD-AI Training & Postprocessing Workflow

Trains and evaluates a BiLSTM + Bahdanau-attention model that forecasts a
vehicle's cellular serving cell 1-7 seconds ahead, from the AI-ready dataset
produced by the [FUMD-AI preprocessing workflow](https://github.com/FUMD-AI/fumd-ai-preprocessing-workflow)
(`dataset_labeled_w<W>.csv`), then postprocesses the results: training-metrics
comparison across runs, real-measurement base-station location estimation,
and spatial mapping of handover/migration events.

## Quick start

Out of the box, `run_pipeline.ipynb` runs Steps 1-4 against a tiny bundled
example slice (`example-data/dataset_labeled_example.csv`, 8 vehicles cut
from a real preprocessing-workflow run) with a small `EPOCHS`, so the whole
chain is runnable immediately:

```
jupyter execute run_pipeline.ipynb
```

For a real dataset, either edit `run_pipeline.ipynb`'s own parameters cell
(`DATASET_PATH`, `EPOCHS`, ...) for local/interactive use, or -- since this
project's actual execution environment submits **one notebook at a time** to
Slurm -- edit and submit the GPU steps individually:

```
# edit notebooks/step_1_load_and_window_dataset.ipynb's parameters cell, then:
jupyter execute notebooks/step_1_load_and_window_dataset.ipynb   # no GPU needed

# edit notebooks/step_2_train_model.ipynb's parameters cell, then:
sbatch slurm/train_model.sbatch                                   # GPU

# edit notebooks/step_3_evaluate_model.ipynb's parameters cell, then:
sbatch slurm/evaluate_model.sbatch                                 # GPU

# edit notebooks/step_4_aggregate_training_metrics.ipynb's parameters cell, then:
jupyter execute notebooks/step_4_aggregate_training_metrics.ipynb  # no GPU needed
```

See "Execution environment" below for why there's no `papermill`-style
single-command orchestration of the GPU steps.

## Pipeline overview

Six notebooks. Steps 1-4 form the core chain from the preprocessing
workflow's labeled dataset to a trained, evaluated model; Steps 5-6 are
optional postprocessing that need your own real-world data (not simulation
output) and no GPU.

| # | Notebook | Purpose | Input (default) | Output (default) | GPU? |
|---|----------|---------|-------|--------|------|
| 1 | `step_1_load_and_window_dataset.ipynb` | Load a `dataset_labeled_w<W>.csv`, encode categoricals, split by vehicle, build sliding-window sequences, fit/apply the feature scaler. | `example-data/dataset_labeled_example.csv` | `windows.npz`, `scaler.joblib`, `label_encoders.joblib`, `run_manifest.json` | no |
| 2 | `step_2_train_model.ipynb` | Build and train the BiLSTM + Bahdanau-attention forecaster on Step 1's windows. | Step 1 output | `model.keras`, `training_history.json`, `training_curves.png` | **yes** |
| 3 | `step_3_evaluate_model.ipynb` | Evaluate on the run's own validation split (+ a `migration`-state accuracy breakdown), and optionally on other datasets for a cross-run generalization check. | Step 2 output | `metrics_<run_label>.csv`/`.json` | yes (predict-only) |
| 4 | `step_4_aggregate_training_metrics.ipynb` | Compare Step 3 metrics across runs/variants -- tables and plots. | one or more Step 3 output dirs | `metrics_long.csv`, `metrics_summary.csv`, comparison plots | no |
| 5 *(optional)* | `step_5_estimate_bs_locations.ipynb` | Estimate real base-station site coordinates from measured RSRP data. **Bring your own data** -- nothing bundled. | your own RSRP CSV | `bs_sites.csv`, `coverage_and_sites.pdf` | no |
| 6 *(optional)* | `step_6_map_migration_events.ipynb` | Plot handover/migration events by type and destination cell over vehicle positions, optionally with real BS coordinates and a street map. | preprocessing's own `dataset_labeled_w<W>.csv` + `events_all_w<W>.csv` | `migration_events_by_type.pdf` | no |

`run_pipeline.ipynb` chains Steps 1-4 for local/interactive use (see
"Execution environment" for why it isn't meant for the Slurm-submitted GPU
steps). Every notebook has a single tagged `parameters` cell and also
stands on its own.

## Execution environment

This project's notebooks run inside a Slurm job, one notebook per job, via:

```
singularity exec --nv --pwd /workflow --bind .:/workflow image_cuda_jupyter.sif \
    jupyter execute /workflow/notebooks/step_2_train_model.ipynb
```

(`slurm/train_model.sbatch` / `slurm/evaluate_model.sbatch` are ready-to-edit
templates for this.) `jupyter execute` is `nbclient`'s own CLI -- it has no
`-p NAME VALUE`-style parameter-injection flag the way `papermill` does, so
a run is configured one of two ways:

1. **Edit the notebook's own `parameters` cell** before submitting it,
   exactly as the original exploratory notebooks were copied and
   hand-edited per dataset; or
2. **Export environment variables** of the same name before calling
   `singularity exec`/`jupyter execute` (see the commented-out examples in
   `slurm/*.sbatch`). Every notebook has a small cell immediately after its
   `parameters` cell that reads any matching environment variables via
   `fumd_training_workflow.notebook_runner.env_override` and applies them,
   leaving the parameters-cell default in place for anything left unset.
   `singularity exec` (these templates don't pass `--cleanenv`) forwards
   the submitting shell's environment into the container, so this works
   the same way under Slurm as it does when calling `jupyter execute`
   directly. Values are JSON-decoded when possible (so
   `SEQUENCE_LENGTH=6`, `FUTURE_STEPS='[1,2,3]'`, etc. round-trip to the
   right type), and used as a plain string otherwise (the common case: a
   bare path like `DATASET_PATH=/abs/path/dataset_labeled_w3.csv`).

Each notebook also normalizes its own working directory at the top of its
imports cell: `jupyter execute <notebook>` runs with that notebook's *own*
containing directory as the kernel's cwd (`notebooks/` for a standalone
step notebook -- not the repository root), so every notebook `chdir`s back
up to the repository root first if it detects it was launched that way.
This is what makes `sys.path.insert(0, "src")` and every relative default
path (`example-data/...`, `pipeline_run`, ...) resolve the same way whether
a notebook is run standalone (`jupyter execute notebooks/step_N...ipynb`,
including every `slurm/*.sbatch` submission) or chained from
`run_pipeline.ipynb`.

`run_pipeline.ipynb` still exists for local/interactive convenience (e.g.
running the whole chain against the bundled example on a laptop, or as a
quick end-to-end smoke test) -- it chains the four core steps via
`src/fumd_training_workflow/notebook_runner.py`, which does the same
"overwrite-the-parameters-cell-then-execute" trick `papermill` is built
around, implemented directly on `nbformat`/`nbclient` (both already required
by `nbconvert`, which `requirements/train.txt`/the `.sif` image includes)
rather than adding a `papermill` dependency that may not be installable
inside a managed HPC image.

## Requirements

Two separate requirement sets, matching the two execution contexts above:

- `requirements/train.txt` -- Steps 1-4, **pinned exactly** to this
  project's `.sif` image (`numpy==1.26.4`, `pandas==2.2.2`,
  `tensorflow==2.14.0`, `keras==2.14.0`, `scikit-learn==1.5.2`, Python
  3.10.9, ...) so a local dev environment behaves identically to the GPU
  cluster.
- `requirements/postprocess.txt` -- Steps 5-6, plain
  pandas/scikit-learn/matplotlib(+`pyrosm` for the optional street-map
  plots), unpinned, meant to run locally rather than through the `.sif`.

```
pip install -r requirements/train.txt          # Steps 1-4
pip install -r requirements/postprocess.txt     # Steps 5-6 (separate environment)
```

## Example data

`example-data/dataset_labeled_example.csv` is a small real slice (8
vehicles, ~1300 rows) cut directly from a real
[FUMD-AI preprocessing workflow](https://github.com/FUMD-AI/fumd-ai-preprocessing-workflow)
run's `dataset_labeled_w3.csv` output -- unlike the preprocessing
workflow's own bundled example (tiny synthetic data, just enough to prove
the pipeline mechanics work), this one is real simulation output, chosen to
include actual `migration` events so Step 1's sliding windows and Step 3's
migration-state accuracy breakdown both have something to show.
`example-data/events_all_example.csv` is the matching slice of that run's
`events_all_w3.csv`, for Step 6.

Steps 5-6 have no bundled example -- see their own "bring your own data"
notes above.

## Repository layout

```
.
├── README.md
├── CITATION.cff                 citation metadata (GitHub/Zenodo citation widget)
├── run_pipeline.ipynb            local/interactive orchestrator: runs Steps 1-4 in one call
├── LICENSE.txt                  MIT (source code)
├── LICENSE-CC-BY-4.0.txt        CC BY 4.0 (explanatory text/figures)
├── ro-crate-metadata.json       FAIR/WorkflowHub packaging metadata
├── requirements/
│   ├── train.txt                Steps 1-4 (pinned to the .sif image)
│   └── postprocess.txt           Steps 5-6 (local, unpinned)
├── slurm/
│   ├── train_model.sbatch        Step 2 Slurm/Singularity submission template
│   └── evaluate_model.sbatch     Step 3 Slurm/Singularity submission template
├── example-data/
│   ├── dataset_labeled_example.csv
│   └── events_all_example.csv
├── src/
│   └── fumd_training_workflow/
│       ├── __init__.py
│       ├── data.py              dataset loading, categorical encoding, sequence windowing, scaling
│       ├── model.py              BahdanauAttention layer + build_model()
│       ├── evaluate.py           metrics: loss/accuracy/precision/recall/F1/top-2/confusion matrix, migration breakdown
│       ├── metrics_io.py         structured (CSV/JSON) metrics read/write
│       ├── bs_location.py        real-measurement BS site estimation (Step 5)
│       ├── migration_map.py      migration-event spatial plotting (Step 6)
│       └── notebook_runner.py    minimal parameter-injection + execution (run_pipeline.ipynb only)
└── notebooks/
    ├── step_1_load_and_window_dataset.ipynb
    ├── step_2_train_model.ipynb
    ├── step_3_evaluate_model.ipynb
    ├── step_4_aggregate_training_metrics.ipynb
    ├── step_5_estimate_bs_locations.ipynb
    └── step_6_map_migration_events.ipynb
```

## Notes on the source notebooks

This workflow rebuilds five one-off exploratory notebooks
(`combined_TimeSequence_*_BahdanauAtention_encoder_2BLSTM.ipynb`,
`leer_metricas.ipynb`, `estimacion_antenas.ipynb`,
`mapa_EB_K_mapAlacant.ipynb`, `mapa_EB_K-mapAveiro.ipynb`) as a parameterized
pipeline against the *new* preprocessing workflow's dataset schema, fixing
several issues found along the way rather than preserving them:

- **Dataset schema.** The exploratory notebooks were written against an
  older, ad hoc CSV format (`newCombinedPast_vehicles_<id>.csv`) that
  predates the `migration`/`destination` labels and included `x-1..x-7`/
  `y-1..y-7` past-position columns the current preprocessing workflow's
  Step 6 deliberately no longer emits (see `data.py`'s module docstring).
  This workflow's feature set (`data.DEFAULT_FEATURE_COLUMNS`) matches what
  `dataset_labeled_w<W>.csv` actually contains today.
- **Lead/future-column leakage risk.** `servingCell1..7` (no dash) in the
  dataset are *lead* (future) values -- easy to mistake for another lagged
  feature. Excluded from the feature set explicitly and documented, not
  just implicitly (the original notebook happened to exclude them too, but
  without comment).
- **Scaler re-fit per dataset.** The original notebook called
  `StandardScaler.fit_transform` on every evaluation dataset, not just the
  training set -- silently making loss/accuracy numbers across datasets
  non-comparable (each was scaled by its own statistics). `data.apply_scaler`
  is transform-only by design; the scaler is fit once, on the training
  split, in Step 1.
- **Label encoder re-fit per file.** Same issue for the categorical `lane`
  column: a fresh `LabelEncoder` per file means the same edge id gets a
  different code in different files. `data.SafeLabelEncoder` is fit once (on
  the full dataset, before the train/val split) and maps any category not
  seen during fitting to one reserved "unknown" code on `transform`, instead
  of raising or silently drifting.
- **Train/validation split by window, not by vehicle.** The original split
  `train_test_split(X_total, y_total, ...)` on the already-built sliding
  windows -- consecutive windows from the same vehicle overlap heavily (they
  share all but one row), so this leaks near-duplicate samples across
  train/validation. `data.split_vehicles` splits *before* windowing, by
  whole vehicle, so no validation window shares a single source row with a
  training window.
- **Copy-pasted per-dataset evaluation.** ~150 lines of near-identical code
  (one block per held-out dataset) became one loop
  (`step_3_evaluate_model.ipynb`'s `CROSS_RUN_DATASET_PATHS`).
- **Regex log-scraping.** `leer_metricas.ipynb` parsed a free-text `.txt`
  log (itself produced by print statements sprinkled through the training
  notebook) back into tables with regular expressions -- fragile and
  format-sensitive. Step 3 writes structured CSV/JSON directly
  (`metrics_io.py`); Step 4 reads that, not prose.
- **Hardcoded `num_base_stations`.** The original notebook hardcoded `10`;
  `evaluate.infer_num_base_stations` derives it from the data (covering
  `servingCell`, its lags, and `destination`), so a run with a different
  base-station count (e.g. Alicante's 9) doesn't silently mismatch the
  model's output-head size.
- **Per-city hardcoded BS coordinates.** `mapa_EB_K_mapAlacant.ipynb` /
  `mapa_EB_K-mapAveiro.ipynb` duplicated almost all of their logic, differing
  mainly in a hardcoded coordinate dict (or a computed one). `migration_map.py`
  takes `bs_coords` as a plain parameter instead.
- **New: migration-state accuracy breakdown.** `evaluate.evaluate_migration_breakdown`
  reports accuracy separately for steady-state vs. pre-handover-window rows
  -- only possible because the new preprocessing pipeline's `migration`
  label didn't exist for the old ad hoc datasets. A single aggregate
  accuracy figure can hide a model that's only good at the (dataset-
  dominant) trivial "predict no change" case.

The model architecture itself (2x bidirectional LSTM encoder + per-step
Bahdanau-attention decoder, `model.py`) is unchanged from the original
notebook -- it's a validated design (~95% top-1 / ~99% top-2 accuracy across
8 real datasets in the original study).

## Validation status

`src/fumd_training_workflow/data.py`, `evaluate.py`, `metrics_io.py`,
`bs_location.py`, and `migration_map.py`, and Steps 1, 3 (its
non-TensorFlow logic; TensorFlow itself was stubbed out for this check),
4, 5, and 6 have each been exercised end-to-end against real data (the
bundled example slice, plus synthetic RSRP/event data for Steps 5-6) in the
course of building this workflow. `model.py` (`build_model`,
`BahdanauAttention`) and Step 2's actual training run have been reviewed
carefully and pass a plain syntax check, but **have not been executed**
end-to-end -- no TensorFlow install was available in the environment this
workflow was built in. Run `run_pipeline.ipynb` (or Steps 1-2 individually)
against the bundled example with a small `EPOCHS` as a first real check
before a full training run.

## Development notes

**v0.1.0** initial release: rebuilds the five exploratory notebooks listed
above as this parameterized pipeline. See "Notes on the source notebooks"
above for the specific issues fixed along the way.

## Authors

- Cristina Bernad, Miguel Hernandez University ([ORCID: 0000-0001-9537-415X](https://orcid.org/0000-0001-9537-415X))
- Sonja Filiposka <sonja.filiposka@finki.ukim.mk>, Ss. Cyril and Methodius University in Skopje ([ORCID: 0000-0003-0034-2855](https://orcid.org/0000-0003-0034-2855))
- Katja Gilly, Miguel Hernandez University ([ORCID: 0000-0002-8985-0639](https://orcid.org/0000-0002-8985-0639))

## Licence

Unless otherwise indicated:

* Source code in this notebook is licensed under the MIT License.

* Explanatory text and original figures are licensed under Creative
  Commons Attribution 4.0 International (CC BY 4.0). Input datasets
  retain the licences stated in their corresponding metadata or
  source records.

SPDX-License-Identifier: MIT

See `LICENSE.txt` (MIT, code) and `LICENSE-CC-BY-4.0.txt` (CC BY 4.0, text/figures).

## Acknowledgement

This work has been funded by the FUMD-AI project, an EOSC GRAVITY - Inter
Project with Grant Number 25-EOSC-GRV-INTER-013.

## Citation

Please cite this workflow if you use it. See `CITATION.cff` and
`ro-crate-metadata.json` for structured citation/author metadata -- a DOI
slot is reserved in both, to be filled in once the workflow is registered
on WorkflowHub.

## FAIR / WorkflowHub packaging

`ro-crate-metadata.json` describes this repository as a
[Workflow RO-Crate](https://w3id.org/workflowhub/workflow-ro-crate/1.0)
(RO-Crate 1.1 + the WorkflowHub workflow profile), the packaging format
WorkflowHub registration expects: `run_pipeline.ipynb` is the crate's main
workflow entity (chaining Steps 1-4), Steps 5-6 are separately described as
their own optional workflow entities, and every notebook parameter is
recorded as a `FormalParameter` with its description and default value, so
the crate stays consistent with each notebook's own `parameters` cell. It
also records authorship/ORCIDs, the FUMD-AI funding grant, licensing, and
the pinned software stack (TensorFlow, scikit-learn, Slurm, Singularity)
this workflow depends on. Regenerate or hand-edit it if the notebooks'
parameters or pipeline structure change -- nothing currently does this
automatically.
