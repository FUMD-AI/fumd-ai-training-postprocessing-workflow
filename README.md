# FUMD-AI Training & Postprocessing Workflow

Trains and evaluates a BiLSTM + Bahdanau-attention model that forecasts a
vehicle's cellular serving cell 1-7 seconds ahead, from the AI-ready dataset
produced by the [FUMD-AI preprocessing workflow](https://github.com/FUMD-AI/fumd-ai-preprocessing-workflow)
(`dataset_labeled_w<W>.csv`), then postprocesses the results: training-metrics
comparison across runs and spatial mapping of handover/migration events.

## Quick start

Out of the box, `run_pipeline.ipynb` runs Steps 1-5 against a tiny bundled
example slice (`example-data/dataset_labeled_example.csv` +
`events_all_example.csv`, 8 vehicles cut from a real preprocessing-workflow
run) with a small `EPOCHS`, so the whole chain -- including the migration-
event map -- is runnable immediately, in one call:

```
jupyter execute run_pipeline.ipynb
```

For a real dataset, edit `run_pipeline.ipynb`'s own parameters cell
(`DATASET_PATH`, `EPOCHS`, ...) and either run it directly for
local/interactive use, or submit the whole chain to Slurm as one job:

```
sbatch slurm/run_pipeline.sbatch   # GPU -- all five steps, one job
```

To run every available dataset run at once -- locally, no Slurm involved,
each one taking a turn as the main dataset with every other run as its
cross-run generalization check, plus a combined-dataset track (trained on
every `_1` run, with the held-out `_2`/`_3` runs as its own cross-run
check -- see "Optional: combining datasets for training (Step 1b)" below)
-- `run_all_datasets.sh` wraps the plain `jupyter execute`
invocation above in a discovery loop over a `datasets/` folder (see
"Execution environment" below for the parallel Slurm version of this same
idea, `slurm/submit_all_runs.sh`):

```
DRY_RUN=1 bash run_all_datasets.sh   # always check this output first
bash run_all_datasets.sh             # then actually run everything
```

Or submit the GPU steps individually instead (finer-grained -- see
"Execution environment" below for the tradeoff between the two):

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

## Pipeline overview

Six notebooks. Steps 1-4 form the core chain from the preprocessing
workflow's labeled dataset to a trained, evaluated model; Step 5 is
optional postprocessing that runs directly against the preprocessing
workflow's own output and needs no GPU; Step 1b is an optional alternate
entry point to Step 1 (combine several runs instead of using just one).

| # | Notebook | Purpose | Input (default) | Output (default) | GPU? |
|---|----------|---------|-------|--------|------|
| 1 | `step_1_load_and_window_dataset.ipynb` | Load a `dataset_labeled_w<W>.csv`, encode categoricals, split by vehicle, build sliding-window sequences, fit/apply the feature scaler. | `example-data/dataset_labeled_example.csv` | `windows.npz`, `scaler.joblib`, `label_encoders.joblib`, `run_manifest.json` | no |
| 2 | `step_2_train_model.ipynb` | Build and train the BiLSTM + Bahdanau-attention forecaster on Step 1's windows. | Step 1 output | `model.keras`, `training_history.json`, `training_curves.png` | **yes** |
| 3 | `step_3_evaluate_model.ipynb` | Evaluate on the run's own validation split (+ a `migration`-state accuracy breakdown), and optionally on other datasets for a cross-run generalization check. | Step 2 output | `metrics_<run_label>.csv`/`.json` | yes (predict-only) |
| 4 | `step_4_aggregate_training_metrics.ipynb` | Compare Step 3 metrics across runs/variants -- tables and plots. | one or more Step 3 output dirs | `metrics_long.csv`, `metrics_summary.csv`, comparison plots | no |
| 5 *(optional)* | `step_5_map_migration_events.ipynb` | Plot handover/migration events by type and destination cell over vehicle positions, optionally with real BS coordinates and a street map (a GeoPackage or a `.osm.pbf` extract). | preprocessing's own `dataset_labeled_w<W>.csv` + `events_all_w<W>.csv` | `migration_events_by_type.pdf` | no |
| 1b *(optional, alternate)* | `step_1b_combine_and_window_datasets.ipynb` | Alternate to Step 1: combine several `dataset_labeled_w<W>.csv` runs into one training set (vehicle ids shifted per source so they can't collide), then window/encode/scale exactly like Step 1. | two or more `dataset_labeled_w<W>.csv` paths (`DATASET_PATHS`, no default) | `windows.npz`, `scaler.joblib`, `label_encoders.joblib`, `run_manifest.json` (same shape as Step 1, in `pipeline_run_combined/` by default) | no |

`run_pipeline.ipynb` chains Steps 1-5 for local/interactive use (see
"Execution environment" for why it isn't meant for the Slurm-submitted GPU
steps). Every notebook has a single tagged `parameters` cell and also
stands on its own.

### Optional: combining datasets for training (Step 1b)

`step_1b_combine_and_window_datasets.ipynb` is a drop-in alternate to Step
1 for training on *several* preprocessing-workflow runs at once instead of
just one -- its output (`windows.npz`/`scaler.joblib`/
`label_encoders.joblib`/`run_manifest.json`) has the exact same shape as
Step 1's, so Step 2 onward need no changes to consume it. It exists mainly
to give the model more real handover/migration examples to learn from
(steady-state rows dominate any single run) -- see the notebook's own
markdown cell, and `data.load_combined_labeled_dataset`'s docstring, for
the full reasoning and for how it prevents different runs' independently-
numbered vehicle ids from colliding when concatenated.

**Deliberately hold out at least one run from `DATASET_PATHS`** -- e.g.
combine only the `_1` runs and keep `_2`/`_3` back -- and point Step 3's
existing `CROSS_RUN_DATASET_PATHS` at those held-out runs afterwards, so
there's still a genuinely unseen run to check generalization against
(Step 3 needs no changes for this, it already accepts a plain list of
paths):

```
DATASET_PATHS='["../datasets/900_1/dataset_labeled_w3.csv","../datasets/1000_1/dataset_labeled_w3.csv","../datasets/1200_1/dataset_labeled_w3.csv"]' \
jupyter execute notebooks/step_1b_combine_and_window_datasets.ipynb
```

Also chainable directly from `run_pipeline.ipynb` in the same call as the
main Steps 1-5 run, as a second, independent track -- set `COMBINE_DATASET_PATHS`
(and, to also evaluate the combined model, `COMBINE_CROSS_RUN_DATASET_PATHS`)
alongside that run's own parameters:

```
DATASET_PATH=../datasets/1000_2/dataset_labeled_w3.csv \
EVENTS_CSV_PATH=../datasets/1000_2/events_all_w3.csv \
OUTPUT_DIR=pipeline_run_1000_2 \
EPOCHS=50 \
CROSS_RUN_DATASET_PATHS='["../datasets/900_1/dataset_labeled_w3.csv","../datasets/1000_1/dataset_labeled_w3.csv","../datasets/1000_3/dataset_labeled_w3.csv","../datasets/1200_1/dataset_labeled_w3.csv","../datasets/1200_2/dataset_labeled_w3.csv","../datasets/1200_3/dataset_labeled_w3.csv"]' \
COMBINE_DATASET_PATHS='["../datasets/900_1/dataset_labeled_w3.csv","../datasets/1000_1/dataset_labeled_w3.csv","../datasets/1200_1/dataset_labeled_w3.csv"]' \
COMBINE_OUTPUT_DIR=pipeline_run_combined \
COMBINE_CROSS_RUN_DATASET_PATHS='["../datasets/1000_2/dataset_labeled_w3.csv","../datasets/1000_3/dataset_labeled_w3.csv","../datasets/1200_2/dataset_labeled_w3.csv","../datasets/1200_3/dataset_labeled_w3.csv"]' \
jupyter execute run_pipeline.ipynb
```

This trains and evaluates *two* independent models in one call -- the main
run's model (on `DATASET_PATH` alone, under `OUTPUT_DIR`) and the combined
model (on `COMBINE_DATASET_PATHS`, under `COMBINE_OUTPUT_DIR`) -- plus maps
for both: the main run's migration map (Step 5, since `EVENTS_CSV_PATH` is
set) and, for the combined track, one migration map *per source run* that
went into `COMBINE_DATASET_PATHS`. The combined track runs Step 1b -> Step
2 -> Step 3 -> Step 4 (metrics comparison across whatever `metrics_*.json`
Step 3 wrote -- train_val plus any `COMBINE_CROSS_RUN_DATASET_PATHS`) just
like the main track does, under `COMBINE_OUTPUT_DIR`/`metrics_comparison/`.
Step 5 is different for this track: there's no single coherent "combined"
positions/events dataset to map (the source runs' vehicle ids were only
shifted to avoid training collisions, not merged into one consistent
simulation), so instead it maps each source dataset in
`COMBINE_DATASET_PATHS` on its own -- under
`COMBINE_OUTPUT_DIR/migration_maps/<run_label>/`, `run_label` being the
source's containing directory name (e.g. `900_1`), reusing the main run's
`BS_COORDS`/`OSM_GEOPACKAGE_PATH`/`OSM_PBF_PATH` overlays and each source's
own `events_all_w<W>.csv` (found automatically next to its
`dataset_labeled_w<W>.csv` via `migration_map.derive_events_path` -- a
source missing its events file has just its own map skipped, not the whole
pipeline). Set `COMBINE_DATASET_PATHS = []` (the default) to skip this
whole track and run only the main Steps 1-5 chain, same as before this was
added. Also still runnable standalone, one step at a time, exactly as
above.

## Execution environment

This project's notebooks run inside a Slurm job via:

```
singularity exec --nv --pwd /workflow --bind .:/workflow image_jupiter_eosc.sif \
    jupyter execute /workflow/notebooks/step_2_train_model.ipynb
```

Two submission patterns are available, and both use exactly this
invocation (just against a different notebook):

- **One notebook per job** -- `slurm/train_model.sbatch` (Step 2) /
  `slurm/evaluate_model.sbatch` (Step 3), the finer-grained option: only
  the GPU-needing steps hold a GPU allocation, and a failed Step 2 can be
  resubmitted without re-running Step 1. Step 1 (windowing) and Step 4
  (metrics aggregation) are meant to run locally/on a login node instead
  (see `requirements/train.txt`'s own comment on this); Step 5 (mapping)
  likewise -- see `requirements/postprocess.txt` and "Notes on the source
  notebooks" below.
- **The whole chain as one job** -- `slurm/run_pipeline.sbatch` submits
  `jupyter execute run_pipeline.ipynb` itself, chaining all five steps
  (including 1/4/5) inside a single GPU allocation, mirroring a local
  `jupyter execute run_pipeline.ipynb` run exactly. This works because
  `run_pipeline.ipynb`'s own kernel spawns each step's kernel as a plain
  OS subprocess (`nbclient.NotebookClient`, invoked from
  `src/fumd_training_workflow/notebook_runner.py`'s `run_step`), which
  inherits this job's environment, Singularity's `--nv` GPU device
  bindings, and Slurm's cgroup-based GPU allocation the same way any other
  child process of the job would -- there's no separate scheduling step
  where a child kernel could land on a different, non-GPU node. The
  tradeoff against the per-step templates: this holds a GPU allocation for
  the whole run, including the CPU-only steps, which wastes GPU-node time
  on a contended queue; simpler to submit and reason about (one job, one
  log) is the upside. This chaining/GPU-inheritance claim has since been
  confirmed on real Slurm infrastructure -- see "Validation status" below.

For running every available dataset run at once (each one taking a turn as
the main `DATASET_PATH`, every other run as its `CROSS_RUN_DATASET_PATHS`
generalization check, plus a combined-dataset track trained on every `_1`
run with the held-out `_2`/`_3` runs as its own
`COMBINE_CROSS_RUN_DATASET_PATHS` check -- see "Optional: combining
datasets for training (Step 1b)" above), `slurm/submit_all_runs.sh` wraps
`run_pipeline.sbatch` in a discovery loop
instead of hand-writing one `export`/`sbatch` invocation per run: it scans
`DATASETS_DIR` for every subfolder containing a `dataset_labeled_w<W>.csv`,
then submits one job per run found (`DRY_RUN=1 bash
slurm/submit_all_runs.sh` prints every `sbatch` invocation it would make
without submitting anything -- always run this first and check the
discovered run list and generated paths, since a mistake here costs real
GPU time across potentially a dozen-plus jobs). See its own header comment
for every override (`DATASETS_DIR`, `W`, `EPOCHS`, `FIRST_RUN_TIME`,
`COMBINE_CROSS_RUN_CHECKS`). Newly added and checked with a synthetic
12-folder dataset layout, not yet run against real data on real Slurm
infrastructure -- confirm with `DRY_RUN=1` first.

`run_pipeline.sbatch` and `evaluate_model.sbatch` -- the two templates
whose notebooks take raw `dataset_labeled_w<W>.csv`/`events_all_w<W>.csv`
paths (`DATASET_PATH`, `EVENTS_CSV_PATH`, `CROSS_RUN_DATASET_PATHS`,
`COMBINE_DATASET_PATHS`, ...) -- bind a second host directory,
`DATASETS_DIR`, into the container at `/datasets`, separately from
`REPO_DIR`'s own bind at `/workflow`. This matters because Singularity's
`--bind` only exposes the exact directory you name, not its parent: this
project's own convention (see the `../datasets/...` paths used throughout
this README's real-data command examples above, e.g. "Optional: combining
datasets for training (Step 1b)") keeps datasets in a folder next to the
repository checkout, not inside it (so it's never committed), and
a relative path like `../datasets/...` from inside `/workflow` wouldn't
resolve to anything actually mounted in the container even though the
identical relative path works fine locally (not inside a container) --
this doesn't depend on whatever a given cluster's Singularity
configuration happens to auto-bind by default, which varies. `DATASETS_DIR`
defaults to `../datasets` (a sibling of `REPO_DIR`, matching that
convention); dataset-path parameters for these two templates are then set
using the in-container `/datasets/...` mount point, not the host path --
see either template's own header comment for worked examples.
`train_model.sbatch` doesn't need this: Step 2 alone only takes `INPUT_DIR`/
`OUTPUT_DIR` (Step 1's windowed output, not a raw dataset file).

All three templates also pass `--env PYTHONNOUSERSITE=1` to `singularity
exec`. None of them pass `--contain`/`--no-home`, so Singularity auto-mounts
the submitting user's host `$HOME` into the container by default (needed
for things like SSH config on some clusters) -- but Python's own
user-site-packages mechanism (`~/.local/lib/pythonX.Y/site-packages`) takes
precedence over a container's system site-packages, so without this flag
anything a user happens to have `pip install --user`-ed on the host can
silently shadow this project's pinned `requirements/train.txt`/
`requirements/postprocess.txt` versions inside the container -- this is
exactly what caused a real `jupyter_client`/`typing_extensions` version
mismatch (a `TypedDict... extra_items` `TypeError` at `jupyter execute`
startup) the first time this pipeline was run on real Slurm infrastructure.
`PYTHONNOUSERSITE=1` disables the user-site lookup entirely, so the
container always uses its own pinned packages regardless of what's on the
host, while leaving the rest of `$HOME`'s auto-mount (and the environment-
variable forwarding described below) untouched.

All three templates also pass `--env XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVVM_DIR}"`.
`image_jupiter_eosc.sif` ships CUDA's *runtime* libraries (cuDNN/cuBLAS/
cuFFT -- training and evaluation both reach a real GPU fine) but not the
CUDA *toolkit*'s `nvvm/libdevice` and `ptxas`, which XLA needs to
JIT-compile certain GPU kernels -- notably Keras's own
`optimizer.apply_gradients` step, which is unconditionally XLA-compiled
regardless of anything in this project's own notebook code. Without this,
Step 2 (training) crashes on its very first step: `libdevice not found at
./libdevice.10.bc` if only `libdevice.10.bc` is missing, or `Failed to
launch ptxas` if `ptxas` is missing too, as in this image -- both were hit
in turn the first time this pipeline was run on real Slurm infrastructure,
after the `PYTHONNOUSERSITE=1` fix above got past the earlier
`typing_extensions` crash. `CUDA_NVVM_DIR` points `XLA_FLAGS` at a
directory containing `<dir>/nvvm/libdevice/libdevice.10.bc` and
`<dir>/bin/ptxas` -- on this cluster these were extracted once from
another CUDA-toolkit-containing image already present (`image_jupyter.sif`'s
CUDA 12.6 install) and copied to a location this project owns
(`resources/cuda_nvvm/`), independent of that other image. The exact
source CUDA version doesn't need to match this image's own CUDA 11.8
runtime: `ptxas` is backward-compatible with older PTX, and
`libdevice.10.bc` is stable IR bitcode -- confirmed working end-to-end (a
full real training + evaluation run, see "Validation status" below) using
CUDA 12.6's copies of both against this CUDA-11.8-targeted TensorFlow
build. Since `$HOME` is auto-mounted (same as `PYTHONNOUSERSITE=1` above),
`CUDA_NVVM_DIR` just needs to be a real path under `$HOME` -- no extra
`--bind` required. The real fix is rebuilding `image_jupiter_eosc.sif`
with the CUDA toolkit's `nvcc`/`nvvm` component included, so this
workaround isn't needed -- see `requirements/train.txt`'s own note on
this for why a specific pip package isn't pinned there yet.

`jupyter execute` is `nbclient`'s own CLI -- it has no
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

`run_pipeline.ipynb` is the same notebook either way -- locally for
interactive/development use (e.g. running the whole chain against the
bundled example on a laptop, or as a quick end-to-end smoke test) and,
via `slurm/run_pipeline.sbatch` above, as a single Slurm submission. It
chains all five steps via
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
- `requirements/postprocess.txt` -- Step 5, plain
  pandas/matplotlib (+ `pyrosm` for the optional street-map plots),
  unpinned, meant to run locally rather than through the `.sif`.

```
pip install -r requirements/train.txt          # Steps 1-4
pip install -r requirements/postprocess.txt     # Step 5 (separate environment)
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
`events_all_w3.csv`, for Step 5. `example-data/alicante_bs_coords.csv` is
the real-world base-station site coordinates `migration_map.ALICANTE_BS_COORDS`
is loaded from (`migration_map.load_bs_coords_from_csv`) -- see "Notes on
the source notebooks" below for its provenance and the reasoning for
packaging it as its own file instead of a Python constant.

## Street-map backgrounds (Step 5)

`plot_events_by_type`'s optional gray street-network background (behind
the colored event scatter points) accepts either of two bring-your-own
data sources -- neither is required, the plot works fine on bare
simulation coordinates without one:

- **`OSM_GEOPACKAGE_PATH`** (recommended): a GeoPackage covering your
  event area, e.g. from [BBBike's extract service](https://extract.bbbike.org/)
  (draw or type a small bounding box, choose `format=geopackage.zip`) --
  BBBike's own `.zip` download works directly, no need to unzip it first.
  Read via `migration_map.load_osm_lines_from_geopackage`, which parses
  the GeoPackage's SQLite + WKB geometry encoding directly with the
  stdlib (`sqlite3` + `struct`) -- **no extra dependency** (no
  `geopandas`/`fiona`/GDAL), and no separate download step for a whole
  country/region the way a `.osm.pbf` extract usually needs, since
  BBBike's service crops to exactly the bounding box you give it.
- **`OSM_PBF_PATH`**: a `.osm.pbf` extract (e.g. from
  [Geofabrik](https://download.geofabrik.de/)), read via `pyrosm`. Kept
  for parity with the original `estimacion_antenas.ipynb`-derived
  approach, but `pyrosm` wasn't straightforward to `pip install` in this
  project's own testing (see "Notes on the source notebooks" below) --
  `OSM_GEOPACKAGE_PATH` is the easier path for most users.

If both are set, `OSM_GEOPACKAGE_PATH` is tried first.

## Repository layout

```
.
├── README.md
├── CITATION.cff                 citation metadata (GitHub/Zenodo citation widget)
├── run_pipeline.ipynb            local/interactive orchestrator: runs Steps 1-5 (+ optional combined-dataset track) in one call
├── run_all_datasets.sh          local, no-Slurm equivalent of slurm/submit_all_runs.sh --
│                                  runs run_pipeline.ipynb once per dataset run under
│                                  DATASETS_DIR, each taking a turn as the main dataset
├── LICENSE.txt                  MIT (source code)
├── LICENSE-CC-BY-4.0.txt        CC BY 4.0 (explanatory text/figures)
├── ro-crate-metadata.json       FAIR/WorkflowHub packaging metadata
├── requirements/
│   ├── train.txt                Steps 1-4 (pinned to the .sif image)
│   └── postprocess.txt           Step 5 (local, unpinned)
├── slurm/
│   ├── train_model.sbatch        Step 2 Slurm/Singularity submission template
│   ├── evaluate_model.sbatch     Step 3 Slurm/Singularity submission template
│   ├── run_pipeline.sbatch       Steps 1-5 chained, one Slurm/Singularity job
│   └── submit_all_runs.sh        submits one run_pipeline.sbatch job per
│                                  dataset run under DATASETS_DIR (each run
│                                  taking a turn as the main dataset, every
│                                  other run as its cross-run check), plus
│                                  the combined-dataset track for the first
│                                  run -- see its own header comment
├── example-data/
│   ├── dataset_labeled_example.csv
│   ├── events_all_example.csv
│   └── alicante_bs_coords.csv    real BS site coordinates -- migration_map.ALICANTE_BS_COORDS's source
├── src/
│   └── fumd_training_workflow/
│       ├── __init__.py
│       ├── data.py              dataset loading, categorical encoding, sequence windowing, scaling
│       ├── model.py              BahdanauAttention layer + build_model()
│       ├── evaluate.py           metrics: loss/accuracy/precision/recall/F1/top-2/confusion matrix, migration breakdown
│       ├── metrics_io.py         structured (CSV/JSON) metrics read/write
│       ├── migration_map.py      migration-event spatial plotting (Step 5)
│       └── notebook_runner.py    minimal parameter-injection + execution (run_pipeline.ipynb only)
└── notebooks/
    ├── step_1_load_and_window_dataset.ipynb
    ├── step_1b_combine_and_window_datasets.ipynb   optional alternate to Step 1: combine several runs
    ├── step_2_train_model.ipynb
    ├── step_3_evaluate_model.ipynb
    ├── step_4_aggregate_training_metrics.ipynb
    └── step_5_map_migration_events.ipynb
```

## Notes on the source notebooks

This workflow rebuilds four one-off exploratory notebooks
(`combined_TimeSequence_*_BahdanauAtention_encoder_2BLSTM.ipynb`,
`leer_metricas.ipynb`, `mapa_EB_K_mapAlacant.ipynb`,
`mapa_EB_K-mapAveiro.ipynb`) as a parameterized pipeline against the *new*
preprocessing workflow's dataset schema, fixing several issues found along
the way rather than preserving them. A fifth exploratory notebook,
`estimacion_antenas.ipynb` (real-measurement base-station location
estimation from RSRP data), was refactored into `step_5_estimate_bs_locations.ipynb`
/ `bs_location.py`, but has since been removed: no real RSRP measurement
data exists anywhere in this project's connected folders to exercise or
validate it against, only a hand-built synthetic fixture, so it was
dropped rather than shipped untested. Its logic remains available in this
repository's git history if real measurement data becomes available later.

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
  takes `bs_coords` as a plain parameter instead -- `migration_map.ALICANTE_BS_COORDS`
  (the same 9-site mapping both original notebooks hardcoded) is Step 5's/
  `run_pipeline.ipynb`'s default, since every dataset bundled with/available
  to this repository was simulated over that same area; pass `{}` or a
  different dict for a different simulated area. That default is itself
  loaded from a bundled file (`example-data/alicante_bs_coords.csv`, via
  `migration_map.load_bs_coords_from_csv`) rather than a Python literal --
  the original notebooks' only recorded provenance was a code comment
  naming a spreadsheet ("Alicante_touristic_places") that isn't itself in
  the repo; as a plain CSV with its own header comment, this coordinate
  data is now findable and reusable as data in its own right (also listed
  in `ro-crate-metadata.json`), not just importable from Python source.
  Adapting to a different simulated area means hand-building an equivalent
  CSV (`cell_id,name,lon,lat`) and loading it the same way -- there's no
  automatic way to derive a simulation cell id's real-world site (see
  `migration_map.py`'s own module docstring).
- **Swapped C2b/C3 event-case labels.** The upstream event data's `case`
  column has "C2b_handover_sin_historico" ("no prior history") and
  "C3_pingpong" ("ABA") with their numeric prefixes swapped (confirmed
  against real `events_all_w3.csv`). `mapa_EB_K_mapAlacant.ipynb` already
  corrected this by hand at plot time; `migration_map.add_case_short` now
  does the same swap once, so every consumer of `case_short` gets the
  corrected code, not just the original notebook's plot.
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

`src/fumd_training_workflow/data.py`, `evaluate.py`, `metrics_io.py`, and
`migration_map.py`, and Steps 1, 4, and 5 have each been exercised
end-to-end against real data (the bundled example slice) in the course of
building this workflow.

`model.py`, Step 2 (training), Step 3 (evaluation), and the full
`run_pipeline.ipynb` chain were originally only reviewed and syntax-checked
-- no TensorFlow install was available in the environment this workflow
was built in -- but have since been run for real (`jupyter execute
run_pipeline.ipynb` on macOS/Apple Silicon, `tensorflow-macos==2.14.0`,
Python 3.10.9) against a full preprocessing run (1,199 vehicles, 123,203
rows, `dataset_labeled_w3.csv`), 5 epochs. Results: top-1 accuracy 95.5%
at +1s degrading to 92.0% at +7s (top-2 accuracy stays above 99% across
the whole horizon), matching the expected pattern of forecasts getting
harder further into the future. The migration-window accuracy breakdown
(a capability the original exploratory notebooks didn't have) shows
exactly the signal it's meant to surface: 97.6% accuracy on steady-state
rows vs. 70.5% in the pre-handover warning window -- the model is
noticeably worse right before a real handover, which is the harder and
more operationally relevant case. This run also caught and fixed four
real bugs the original build/test pass (lacking any TensorFlow
environment) couldn't have caught: a `jupyter execute` working-directory
bug affecting every standalone step-notebook invocation (see "Execution
environment" above), `inject_parameters()` generating invalid Python
for `None`/boolean parameter values, `inject_parameters()` silently
undefining any parameter a caller didn't explicitly override, and a
`Lambda`-layer model architecture choice that Keras's `.keras` safe-mode
deserialization refuses to reload (`model.py` now uses a proper
subclassed `ZeroInitialState` layer instead). Step 5's own migration-map
plotting logic (`migration_map.py`, including the new GeoPackage street-map
path) was then verified against real data too -- a BBBike GeoPackage
extract and the real matched events from the `1000_1` run -- but only by
exec()ing the notebook's actual cell source in sequence, not a genuine
`jupyter execute`/papermill run, since no such tool was available in that
verification environment.

That gap has since been closed: a full `jupyter execute run_pipeline.ipynb`
run (same macOS/Apple Silicon, `tensorflow-macos==2.14.0` environment,
`EPOCHS=50`) against the same full `1200_1` preprocessing run (123,203
rows), with `OSM_GEOPACKAGE_PATH` also set this time, chained Steps 1-5
for real end to end -- zero errors in any of the five executed notebooks.
`EarlyStopping` did exactly what it's meant to: `run_manifest.json` records
`epochs_run: 8` against the `epochs_requested: 50` ceiling, and
`training_curves.png` shows why -- validation loss bottoms out at epoch 4
(0.987) and rises for the following three epochs while training loss keeps
falling, patience=3 calls time at epoch 8, and `restore_best_weights=True`
reloads the epoch-4/5 checkpoint. The final metrics landed almost exactly
where the earlier 5-epoch run did (95.5%/92.0% top-1 at +1s/+7s here vs.
95.53%/92.01% then; 97.6%/70.5% steady-state/warning-window here vs.
97.58%/70.53% then) -- reassuring evidence that the model was already near
its plateau by epoch 5 and the original 5-epoch numbers weren't an
undertrained fluke, rather than a sign anything is wrong with early
stopping. Step 5's map (chained, not standalone) now also carries the real
street-map background -- `OSM_GEOPACKAGE_PATH` threading through
`run_pipeline.ipynb`'s own parameters correctly on a genuine run, matching
the exec()-based check that first caught the gap -- and every cell in all
six notebooks now round-trips through real execution with its `id` field
intact (nbformat's `MissingIDFieldWarning` is gone).

Step 3's cross-run generalization check (`CROSS_RUN_DATASET_PATHS`) has
also now been run for real, standalone against the already-trained
`1200_1` model (no retraining needed -- Step 3 only transforms with the
fitted scaler/encoders, never refits them) across every other full
preprocessing run available (`1000_1`, `1000_2`, `1000_3`, `1200_2`,
`1200_3`, `900_1` -- `900_2`/`900_3` excluded, 17-row incomplete runs).
Result: the model generalizes cleanly -- overall accuracy on all six
unseen runs (94.3%-94.6%) is actually slightly *higher* than on its own
held-out `train_val` split (93.9%), and every run shows the same
per-forecast-step degradation shape from +1s to +7s, no outliers or
collapsed/degenerate scores. That's a good sign against overfitting to
one run's specific road network or traffic pattern, though it's also
only six runs from what looks like the same underlying simulated area at
different vehicle counts -- not evidence of generalizing to a
genuinely different road network or city. It's also currently
uneven across vehicle-density tiers -- `900_2`/`900_3` weren't available
as complete runs at the time of this check, so there's no held-out
coverage for the 900-vehicle tier specifically (only as the `900_1`
*training* run); once real `900_2`/`900_3` data exists (a `1400`-vehicle
tier is also expected later), re-running this check against them would
close that gap.

The new optional `step_1b_combine_and_window_datasets.ipynb` (combines
several runs into one training set via `data.load_combined_labeled_dataset`,
shifting each source's vehicle ids so they can't collide) was first
verified the exec()-based way against real `900_1`/`1000_1`/`1200_1`
data -- 900+999+1199=3,098 combined vehicles, zero cross-source id
collisions, correct per-source counts through encoding/splitting/
windowing/scaling/saving -- and has since also been run for real
(`jupyter execute notebooks/step_1b_combine_and_window_datasets.ipynb`
against the same three runs), zero errors. `run_manifest.json` from that
real run reports 223,410 train windows and 55,323 val windows -- an exact
match to the earlier exec()-based check, confirming the two runs agree
(deterministic given `RANDOM_SEED=42`).

Step 2 has since also been run for real against that combined output
(`INPUT_DIR=pipeline_run_combined OUTPUT_DIR=pipeline_run_combined jupyter
execute notebooks/step_2_train_model.ipynb`), zero errors, `EPOCHS=50`
default. `EarlyStopping` behaved as expected: `run_manifest.json` records
`epochs_run: 6` against the 50-epoch ceiling (`val_loss` bottoms at epoch
3, rises for `EARLY_STOPPING_PATIENCE=3` epochs after,
`restore_best_weights=True` reloads the epoch-3 checkpoint). Per-output
accuracy at the final logged epoch runs 96.0% (val) at +1s down to 93.0%
at +7s -- the same degradation shape from +1s to +7s as the earlier
single-run (`1200_1`) baseline. Step 3 has since evaluated this model too
-- see below.

The C2b/C3 case-label swap and the `ALICANTE_BS_COORDS` default (see
"Notes on the source notebooks" above) have also both been confirmed by a
genuine `jupyter execute notebooks/step_5_map_migration_events.ipynb` run
against real `1200_1` data with `OSM_GEOPACKAGE_PATH` set -- zero errors,
and the resulting `migration_events_by_type.pdf` matches
`migration_events_4928.pdf`'s reference layout (BS rings in the right
places/colors, C2b denser than C3 as expected for ABA vs. no-prior-history
events).

`run_pipeline.ipynb`'s combined-dataset track
(`COMBINE_DATASET_PATHS`/`COMBINE_OUTPUT_DIR`/`COMBINE_ID_OFFSET`/
`COMBINE_CROSS_RUN_DATASET_PATHS`) has since been run for real -- and for
the first time together with the main track in one invocation: a single
`jupyter execute run_pipeline.ipynb` call with `DATASET_PATH` pointing at
`1000_2` (main Steps 1-5 track, `OSM_GEOPACKAGE_PATH` set) and
`COMBINE_DATASET_PATHS` set to `900_1`/`1000_1`/`1200_1` (combined track,
with its own `COMBINE_CROSS_RUN_DATASET_PATHS`) -- zero errors across all
eight executed notebooks (the main track's five plus the combined track's
Step 1b/2/3). This also exercises `notebook_runner.run_step`'s env-var
pop/restore behavior for real: both tracks set same-named parameters
(`OUTPUT_DIR`, `CROSS_RUN_DATASET_PATHS`) to different values in the same
process, and neither leaked into the other.

The main-track model (trained on `1000_2` alone -- 70,922 train / 16,860 val
windows, `epochs_run: 10`) scored 93.9% train/val accuracy overall (95.2% at
+1s, 92.5% at +7s), with the same steady-state-vs-warning-window split seen
in earlier runs (97.3% vs. 70.0%), and generalized to the six other full
preprocessing runs (`900_1`, `1000_1`, `1000_3`, `1200_1`, `1200_2`,
`1200_3`) at 93.0%-94.4% accuracy -- consistent with its own held-out split,
no outliers. `migration_events_by_type.pdf` (Step 5, this time against
`1000_2`) again shows the C2b/C3 fix and BS-coordinate rings correctly.

The combined-track model (trained on `900_1`+`1000_1`+`1200_1` --
223,410 train / 55,323 val windows, `epochs_run: 6`, matching the earlier
standalone Step 1b run's window counts exactly) scored higher across the
board: 95.1% train/val accuracy (96.3% at +1s, 93.7% at +7s), and
94.9%-95.1% on the four runs it never trained on (`1000_2`, `1000_3`,
`1200_2`, `1200_3`) -- a bit above the single-run model's equivalent numbers
in every case. That's the first direct evidence in this workflow that
training on combined runs helps generalization, not just adds more of the
same data.

That run predated two further changes to the combined track, made once the
gap was noticed (see "Optional: combining datasets for training (Step 1b)"
above): Step 4 (metrics comparison) and Step 5 (one migration map per source
run in `COMBINE_DATASET_PATHS`, via the new `migration_map.derive_events_path`
helper) chained after Step 3, and Step 3 itself made to always run
(previously it was skipped entirely -- along with the combined model's own
`metrics_train_val.json` -- whenever `COMBINE_CROSS_RUN_DATASET_PATHS` was
left empty; the real run above happened to set it, so this bug was latent
rather than triggered there). All three changes have since been confirmed
for real too: a second full `jupyter execute run_pipeline.ipynb` run, same
parameters as above (main track against `1000_2`, combined track against
`900_1`/`1000_1`/`1200_1`, both with their cross-run checks), zero errors
across all twelve executed notebooks -- the main track's five, plus the
combined track's Step 1b/2/3/4 and three per-source Step 5 maps
(`pipeline_run_combined/migration_maps/900_1/`, `.../1000_1/`,
`.../1200_1/`). Every window count, epoch count, and accuracy number
matched the first run exactly (deterministic given `RANDOM_SEED=42`),
confirming the new Step 4/5 chaining and the Step 3 fix didn't change any
model behavior -- only added the missing outputs. `metrics_comparison/`
under `pipeline_run_combined/` now has the same three comparison plots +
summary/long CSVs the main track's Step 4 always produced, and each
per-source map shows the correct BS rings and C2b/C3 pattern, same as the
main track's own map.

This same run also confirmed `migration_map.ALICANTE_BS_COORDS`'s new
CSV-backed loading (`load_bs_coords_from_csv` reading
`example-data/alicante_bs_coords.csv`, see "Notes on the source notebooks"
above): both the main track's map (against `1000_2`) and the combined
track's three per-source maps show BS rings in the identical positions as
before this change, confirming the CSV round-trips to the exact same 9
sites as the old Python literal did.

Every run described above ran on macOS/Apple Silicon (`tensorflow-macos`),
not the actual target Slurm/Singularity/GPU infrastructure. That gap is
now closed: `run_pipeline.sbatch` (main track only, `DATASET_PATH`
pointing at `1000_1`, no `COMBINE_DATASET_PATHS`, `EPOCHS=50`) has been
submitted and run to completion for real on the project's Slurm cluster,
via `image_jupiter_eosc.sif` on an A100 (MIG 7g.80gb slice) GPU node --
confirming the `run_pipeline.ipynb`-as-single-job pattern's GPU-inheritance
claim above holds in practice, not just in theory. Zero errors across all
five executed notebooks, every expected artifact present (`model.keras`,
`scaler.joblib`, `label_encoders.joblib`, `run_manifest.json`,
`metrics_train_val.json`/`.csv`, `training_curves.png`,
`metrics_comparison/`, `migration_maps/migration_events_by_type.pdf`).
`EarlyStopping` stopped at `epochs_run: 10` (of the 50 requested), overall
accuracy 93.9% (95.4% at +1s degrading to 91.9% at +7s, top-2 accuracy
above 98.9% throughout), and the migration-window breakdown again shows
the expected pattern -- 97.6% on steady-state rows vs. 66.7% in the
pre-handover warning window -- consistent with every earlier local run.
The per-step templates (`train_model.sbatch`/`evaluate_model.sbatch`)
haven't separately been submitted on real Slurm yet, but exercise the
identical `singularity exec ... jupyter execute <notebook>.ipynb`
invocation against the same image, just one step and one job at a time,
so nothing about this confirmation is specific to the chained path.

Getting here surfaced two real infrastructure problems this project's
local/macOS testing couldn't have caught, since neither exists outside a
Singularity/Slurm environment: the `PYTHONNOUSERSITE=1` fix (host
`~/.local` package leakage via Singularity's default `$HOME` auto-mount)
and the `CUDA_NVVM_DIR`/`XLA_FLAGS` fix (this image's CUDA toolkit gap --
`libdevice`/`ptxas` both missing) -- see "Execution environment" above for
the full explanation of both. Both are now baked into all three
`slurm/*.sbatch` templates by default.

## Development notes

**v0.1.4** (2026-08-28): the first genuine end-to-end run on real Slurm/
Singularity/GPU infrastructure (`run_pipeline.sbatch`, main track only,
against `1000_1`) -- zero errors, every expected artifact produced, and
accuracy numbers consistent with every earlier local run (93.9% overall,
95.4%/91.9% at +1s/+7s, 97.6%/66.7% steady-state/warning-window); see
"Validation status" above for the full numbers. Getting there surfaced and
fixed two real infrastructure problems local/macOS testing couldn't have
caught: `PYTHONNOUSERSITE=1` (host `~/.local` packages, auto-mounted via
Singularity's default `$HOME` mount, were shadowing this project's pinned
`jupyter_client`/`typing_extensions` versions, causing a `TypedDict...
extra_items` `TypeError` at `jupyter execute` startup) and
`CUDA_NVVM_DIR`/`XLA_FLAGS` (`image_jupiter_eosc.sif` ships CUDA's runtime
libraries but not the toolkit's `nvvm/libdevice`/`ptxas`, which XLA needs
to JIT-compile Keras's own `optimizer.apply_gradients` step, crashing Step
2's very first training step). Also corrected `DATASETS_DIR`'s default in
`run_pipeline.sbatch`/`evaluate_model.sbatch`: the sibling-of-repo
`../datasets` convention didn't match this cluster's actual layout
(`resources/datasets/`), so it now points there directly, same pattern as
`IMAGE`. All three fixes are documented in full in "Execution environment"
above and baked into all three `slurm/*.sbatch` templates by default.

**v0.1.3** (2026-08-28): new optional
`notebooks/step_1b_combine_and_window_datasets.ipynb` (alternate to Step 1:
combines several `dataset_labeled_w<W>.csv` runs into one larger training
set, via the new `data.load_combined_labeled_dataset`) and, chained onto it,
a full second track inside `run_pipeline.ipynb` -- Step 1b -> 2 -> 3 -> 4 -> 5
-- that trains and evaluates an independent "combined" model alongside the
main run in the same call (`COMBINE_DATASET_PATHS`/`COMBINE_OUTPUT_DIR`/
`COMBINE_ID_OFFSET`/`COMBINE_CROSS_RUN_DATASET_PATHS`; empty/default skips
it entirely). Step 5 for this track maps each *source* run in
`COMBINE_DATASET_PATHS` individually (via the new
`migration_map.derive_events_path`), since there's no single coherent
"combined" positions/events dataset to map. Also fixed a real bug found
while wiring Step 4/5 in: Step 3 -- and therefore the combined model's own
held-out metrics -- used to be skipped entirely whenever
`COMBINE_CROSS_RUN_DATASET_PATHS` was left empty; it's now unconditional,
matching the main track. See "Optional: combining datasets for training
(Step 1b)" above for the full behavior and examples.

Also this release: fixed the swapped C2b/C3 event-case labels
(`migration_map.add_case_short`) and added `migration_map.ALICANTE_BS_COORDS`
as Step 5's/`run_pipeline.ipynb`'s default `BS_COORDS`, so migration maps
show base-station markers out of the box; that coordinate data is now
loaded from a bundled, citable file (`example-data/alicante_bs_coords.csv`,
via the new `migration_map.load_bs_coords_from_csv`) instead of being an
anonymous Python literal, closing a real FAIR-packaging gap (also listed in
`ro-crate-metadata.json`) -- see "Notes on the source notebooks" above for
both. `slurm/train_model.sbatch`/`evaluate_model.sbatch`/`run_pipeline.sbatch`
now default `IMAGE` to this project's actual built `.sif` path on the
cluster instead of a bare filename placeholder.

All of the above -- including the combined-dataset track's Step 4/5
chaining, the Step 3 fix, and the CSV-based `ALICANTE_BS_COORDS` loading --
has now been confirmed by a single genuine `jupyter execute
run_pipeline.ipynb` run chaining both tracks together (main track against
`1000_2`, combined track against `900_1`+`1000_1`+`1200_1`, both with cross-
run generalization checks): zero errors across all twelve executed
notebooks (the main track's five, plus the combined track's Step 1b/2/3/4
and three per-source Step 5 maps). Window counts, epoch counts, and every
accuracy number matched the prior verification run exactly (deterministic
given `RANDOM_SEED=42`), confirming the new orchestration code didn't
change any model behavior -- only added the missing steps. See "Validation
status" below for the full numbers.

Two more fixes came out of the first real submissions on actual Slurm
infrastructure, both in `slurm/*.sbatch`: `run_pipeline.sbatch`/
`evaluate_model.sbatch` now bind a separate `DATASETS_DIR` host directory
into the container at `/datasets`, since a relative `../datasets/...` path
(this project's own convention for keeping datasets outside the repo
checkout) doesn't resolve from inside a container whose only bind is the
repo itself -- Singularity's `--bind` only exposes the exact directory you
name, not its parent; and all three templates now pass `--env
PYTHONNOUSERSITE=1` to `singularity exec`, after a real run hit a
`jupyter_client`/`typing_extensions` version mismatch
(`TypedDict... extra_items` `TypeError` at `jupyter execute` startup)
caused by the submitting user's host `~/.local` packages -- auto-mounted
into the container along with the rest of `$HOME` -- silently shadowing
this project's pinned dependencies via Python's user-site-packages
precedence. See "Execution environment" above for the full explanation of
both.

**v0.1.2** (2026-08-28): `plot_events_by_type` now crops every subplot to
the bounding box of the actual matched events (`zoom_to_events`, default
on, padded 10% by `zoom_padding_frac`) instead of autoscaling to the full
street-map/simulation extent -- the event scatter used to end up crowded
into a small corner of an otherwise-empty plot. `run_pipeline.ipynb` was
also silently dropping `BS_COORDS`/`OSM_GEOPACKAGE_PATH`/`OSM_PBF_PATH`
when chaining Step 5 -- its own parameters cell never had them, so a real
street-map file never reached the map even when set as an environment
variable, no error or warning either. Both fixed and verified together in
one genuine `jupyter execute run_pipeline.ipynb` run -- that run also
confirmed nbformat's `MissingIDFieldWarning` is gone after backfilling
every cell's missing `id` field across all six notebooks (a
future-nbformat-version hard error otherwise, harmless today).

That same real-run-confirmation gap in "Validation status" is now fully
closed: a genuine `jupyter execute run_pipeline.ipynb` run against the
full `1200_1` preprocessing run (123,203 rows), `EPOCHS=50`, confirmed
`EarlyStopping` stops training at the right point (`epochs_run: 8` against
the ceiling, validation loss bottoming at epoch 4 before rising) and
produced accuracy numbers matching the earlier 5-epoch run to within 0.03
points across every metric checked -- see "Validation status" for the
full numbers. Step 3's `CROSS_RUN_DATASET_PATHS` generalization check was
also run for real (standalone, no retraining needed) against every other
full preprocessing run available -- all six scored 94.3%-94.6% accuracy,
slightly *higher* than the model's own held-out validation split (93.9%),
no outliers.

Added `slurm/run_pipeline.sbatch`: submits the full Steps 1-5 chain as a
single Slurm job, the same `jupyter execute run_pipeline.ipynb` invocation
as a local run. This corrects an earlier design claim (in
`run_pipeline.ipynb`'s own markdown cell and this README) that chaining
all five steps as one Slurm job wasn't GPU-reachable -- reasoning through
how Singularity's `--nv` binding and Slurm's GPU allocation scope to a
job's whole process tree (not just its top process) shows a child kernel
spawned by `run_pipeline.ipynb`'s own kernel does inherit that job's GPU
access, the same as any other child process would. Neither Slurm
submission pattern (this new one or the pre-existing per-step templates)
has actually been run on real Slurm infrastructure yet -- see "Validation
status" and `run_pipeline.sbatch`'s own header comment.

Also: `.gitignore` now covers `*.gpkg`/`*.geopackage.zip` alongside the
existing `*.osm.pbf`, so a real GeoPackage extract used for local testing
(as this release's own verification runs did) doesn't get committed by
accident.

**v0.1.1** (2026-08-27): first real (non-syntax-check-only) TensorFlow
execution of `model.py`/Step 2/Step 3/`run_pipeline.ipynb`, against a full
preprocessing run rather than only the bundled example. Fixed four bugs
this surfaced -- see "Validation status" above for what and why. No
architectural or parameter changes beyond the `Lambda` -> `ZeroInitialState`
layer swap (models trained with v0.1.0 are not loadable with this version;
retrain). Also dropped the real-measurement base-station-location-estimation
step (`step_5_estimate_bs_locations.ipynb` / `bs_location.py`) and
renumbered the migration-event-mapping step from Step 6 to Step 5 -- see
"Notes on the source notebooks" for why. `run_pipeline.ipynb` now chains
Step 5 too (controlled by its own `EVENTS_CSV_PATH` parameter, skipped if
unset) so a full local test run -- including the migration-event map --
is one `jupyter execute run_pipeline.ipynb` call instead of five separate
ones. That chaining surfaced two more real bugs, both fixed: (1)
`plot_events_by_type`'s x-axis tick labels overlapped for real-world
lon/lat coordinates (many decimal digits in a narrow, `axis("equal")`
-shrunk subplot) -- fixed by capping the tick count and rotating labels
45deg; (2) `notebook_runner.run_step` didn't isolate each chained child
notebook's environment, so an ambient env var set for
`run_pipeline.ipynb`'s own top-level parameter (e.g. `OUTPUT_DIR`) leaked
into every child step's own same-named `env_override` call and silently
overrode the step-specific value `run_pipeline.ipynb` had just computed
for it (Step 5's migration maps landed directly in `OUTPUT_DIR/` instead
of `OUTPUT_DIR/migration_maps/`) -- fixed by temporarily unsetting exactly
the names each `run_step` call injects, for the duration of that child's
execution.

Also, ahead of a first real Slurm submission: corrected the placeholder
`.sif` filename in `slurm/*.sbatch`/`requirements/train.txt`/README/
ro-crate-metadata.json to the real image, `image_jupiter_eosc.sif`, and
cross-checked its confirmed package list against every import Steps 1-4
actually make. `joblib` isn't separately listed but ships as a hard
scikit-learn dependency, so it's fine; `seaborn` (Step 4's comparison
plots only) genuinely isn't in the image and isn't a dependency of
anything else there, so `step_4_aggregate_training_metrics.ipynb`'s
`comparison_plot` is rewritten on plain matplotlib instead (verified
against both synthetic data and the real `1000_1` run's metrics --
identical faceted/grouped bar charts, no seaborn import anywhere in
`requirements/train.txt`'s dependency closure now). Steps 2-3 (the only
ones that actually run inside the .sif, per `slurm/*.sbatch`) never
depended on seaborn in the first place.

Also added a second, dependency-free way to get a real street-map
background for Step 5's event maps: `migration_map.
load_osm_lines_from_geopackage` reads a GeoPackage (e.g. from BBBike's
extract service) directly via stdlib `sqlite3` + a minimal WKB parser --
no `geopandas`/`fiona`/GDAL needed, unlike the existing `pyrosm`+
`.osm.pbf` path (`OSM_PBF_PATH`), which wasn't straightforward to
`pip install` earlier in this project's own testing (see "conda install
-c conda-forge pyrosm worked" above). See "Street-map backgrounds
(Step 5)" for both options; verified against a real BBBike GeoPackage
extract of the `1000_1` run's event area (659 road segments after
filtering to driving-relevant `highway` tags) and the actual matched
events -- clean overlay, no change needed to the x-axis tick fix above.

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

We gratefully acknowledge Polish high-performance computing infrastructure
PLGrid (HPC Centers: ACK Cyfronet AGH) for providing computer facilities and
support within computational grant no. PLGINT/2026/019844.

The research work was supported by the Open Science Cloud research laboratory
(OSC-LAB) at the Faculty of Computer Science and Engineering (FINKI), Ss.
Cyril and Methodius University in Skopje, North Macedonia.

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
workflow entity (chaining Steps 1-5 -- Step 5 optionally, when
`EVENTS_CSV_PATH` is set), and every notebook parameter is
recorded as a `FormalParameter` with its description and default value, so
the crate stays consistent with each notebook's own `parameters` cell. It
also records authorship/ORCIDs, the FUMD-AI funding grant, licensing, and
the pinned software stack (TensorFlow, scikit-learn, Slurm, Singularity)
this workflow depends on. Regenerate or hand-edit it if the notebooks'
parameters or pipeline structure change -- nothing currently does this
automatically.
