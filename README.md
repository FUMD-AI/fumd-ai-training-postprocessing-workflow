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

Five notebooks. Steps 1-4 form the core chain from the preprocessing
workflow's labeled dataset to a trained, evaluated model; Step 5 is
optional postprocessing that runs directly against the preprocessing
workflow's own output and needs no GPU.

| # | Notebook | Purpose | Input (default) | Output (default) | GPU? |
|---|----------|---------|-------|--------|------|
| 1 | `step_1_load_and_window_dataset.ipynb` | Load a `dataset_labeled_w<W>.csv`, encode categoricals, split by vehicle, build sliding-window sequences, fit/apply the feature scaler. | `example-data/dataset_labeled_example.csv` | `windows.npz`, `scaler.joblib`, `label_encoders.joblib`, `run_manifest.json` | no |
| 2 | `step_2_train_model.ipynb` | Build and train the BiLSTM + Bahdanau-attention forecaster on Step 1's windows. | Step 1 output | `model.keras`, `training_history.json`, `training_curves.png` | **yes** |
| 3 | `step_3_evaluate_model.ipynb` | Evaluate on the run's own validation split (+ a `migration`-state accuracy breakdown), and optionally on other datasets for a cross-run generalization check. | Step 2 output | `metrics_<run_label>.csv`/`.json` | yes (predict-only) |
| 4 | `step_4_aggregate_training_metrics.ipynb` | Compare Step 3 metrics across runs/variants -- tables and plots. | one or more Step 3 output dirs | `metrics_long.csv`, `metrics_summary.csv`, comparison plots | no |
| 5 *(optional)* | `step_5_map_migration_events.ipynb` | Plot handover/migration events by type and destination cell over vehicle positions, optionally with real BS coordinates and a street map (a GeoPackage or a `.osm.pbf` extract). | preprocessing's own `dataset_labeled_w<W>.csv` + `events_all_w<W>.csv` | `migration_events_by_type.pdf` | no |

`run_pipeline.ipynb` chains Steps 1-5 for local/interactive use (see
"Execution environment" for why it isn't meant for the Slurm-submitted GPU
steps). Every notebook has a single tagged `parameters` cell and also
stands on its own.

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
  log) is the upside. Neither pattern has been exercised on real Slurm
  infrastructure yet -- see "Validation status" below.

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
`events_all_w3.csv`, for Step 5.

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
├── run_pipeline.ipynb            local/interactive orchestrator: runs Steps 1-5 in one call
├── LICENSE.txt                  MIT (source code)
├── LICENSE-CC-BY-4.0.txt        CC BY 4.0 (explanatory text/figures)
├── ro-crate-metadata.json       FAIR/WorkflowHub packaging metadata
├── requirements/
│   ├── train.txt                Steps 1-4 (pinned to the .sif image)
│   └── postprocess.txt           Step 5 (local, unpinned)
├── slurm/
│   ├── train_model.sbatch        Step 2 Slurm/Singularity submission template
│   ├── evaluate_model.sbatch     Step 3 Slurm/Singularity submission template
│   └── run_pipeline.sbatch       Steps 1-5 chained, one Slurm/Singularity job
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
│       ├── migration_map.py      migration-event spatial plotting (Step 5)
│       └── notebook_runner.py    minimal parameter-injection + execution (run_pipeline.ipynb only)
└── notebooks/
    ├── step_1_load_and_window_dataset.ipynb
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
genuinely different road network or city.

Neither Slurm submission pattern has been exercised on real Slurm
infrastructure yet -- the per-step templates (`train_model.sbatch`/
`evaluate_model.sbatch`) or the single-job `run_pipeline.sbatch` (see
"Execution environment" above for both). `run_pipeline.sbatch` in
particular rests on a specific claim -- that a child kernel spawned by
`run_pipeline.ipynb`'s own kernel inherits this job's GPU allocation --
that follows from how Singularity/Slurm scope GPU access to a job's whole
process tree, but was reasoned through rather than watched happen on a
real cluster; worth confirming with a first real submission (e.g. a
low-`EPOCHS` run) before relying on it for a real training run.

## Development notes

**v0.1.1** (unreleased): first real (non-syntax-check-only) TensorFlow
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

Also: `plot_events_by_type` now crops every subplot to the bounding box of
the actual matched events (`zoom_to_events`, default on, padded 10% by
`zoom_padding_frac`) instead of autoscaling to the full street-map/
simulation extent -- the event scatter used to end up crowded into a small
corner of an otherwise-empty plot. And `run_pipeline.ipynb` was silently
dropping `BS_COORDS`/`OSM_GEOPACKAGE_PATH`/`OSM_PBF_PATH` when chaining
Step 5 -- its own parameters cell never had them, so a real street-map file
never reached the map even when set as an environment variable, no error
or warning either. Both fixed and verified together in one genuine
`jupyter execute run_pipeline.ipynb` run (see "Validation status" above)
-- that run also confirmed nbformat's `MissingIDFieldWarning` is gone
after backfilling every cell's missing `id` field across all six
notebooks (a future-nbformat-version hard error otherwise, harmless today).

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
workflow entity (chaining Steps 1-5 -- Step 5 optionally, when
`EVENTS_CSV_PATH` is set), and every notebook parameter is
recorded as a `FormalParameter` with its description and default value, so
the crate stays consistent with each notebook's own `parameters` cell. It
also records authorship/ORCIDs, the FUMD-AI funding grant, licensing, and
the pinned software stack (TensorFlow, scikit-learn, Slurm, Singularity)
this workflow depends on. Regenerate or hand-edit it if the notebooks'
parameters or pipeline structure change -- nothing currently does this
automatically.
