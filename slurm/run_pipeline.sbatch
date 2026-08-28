#!/bin/bash
#
# Submits the FULL chain (Steps 1-5, via run_pipeline.ipynb) to Slurm as a
# single GPU job -- the same `singularity exec ... jupyter execute
# run_pipeline.ipynb` invocation as a local/interactive run, just run on a
# cluster node instead of a laptop.
#
# This project's other two templates (train_model.sbatch,
# evaluate_model.sbatch) instead submit ONE step at a time, and
# run_pipeline.ipynb's own markdown cell used to say chaining all five
# steps as one Slurm job wasn't GPU-reachable -- that turns out not to be
# a real technical limit: run_pipeline.ipynb's own kernel spawns each
# step's kernel as a plain OS subprocess (nbclient's NotebookClient), which
# inherits this job's environment, its Singularity `--nv` GPU device
# bindings, and Slurm's own cgroup-based GPU allocation the same way any
# other child process of this job would -- there's no separate scheduling
# step where a child kernel could end up on a different, non-GPU node. See
# README's "Execution environment" section for the full explanation and
# the tradeoff against the still-available per-step templates (this job
# holds a GPU allocation for the whole run, including the CPU-only Steps
# 1/4/5 -- wasteful of GPU-node time if your queue is contended; the
# per-step templates let you skip that for those steps, and let you
# re-submit just Step 2 if it fails without re-running everything before
# it). Genuinely UNTESTED on real Slurm as of this writing -- if your
# cluster enforces a one-notebook-per-job policy independent of the GPU
# question above, use the per-step templates instead.
#
# Before submitting:
#   1. Set this run's parameters (DATASET_PATH, EVENTS_CSV_PATH, EPOCHS,
#      OUTPUT_DIR, CROSS_RUN_DATASET_PATHS, ...) either by editing
#      run_pipeline.ipynb's own "parameters" cell directly, or by exporting
#      environment variables of the same name below -- singularity exec (no
#      --cleanenv here) passes the submitting shell's environment through
#      into the container and, from there, into every chained step's own
#      env-override cell (see src/fumd_training_workflow/notebook_runner.py's
#      env_override() and run_step() docstrings -- run_step() also
#      temporarily unsets each parameter it explicitly injects so an
#      ambient env var like OUTPUT_DIR can't leak into a child step's own
#      differently-scoped parameter of the same name). Optionally also set
#      COMBINE_DATASET_PATHS (and COMBINE_OUTPUT_DIR/COMBINE_ID_OFFSET/
#      COMBINE_CROSS_RUN_DATASET_PATHS) to additionally run Step 1b's
#      combine-multiple-datasets track as a second, independent model in
#      this same job -- see README's "Optional: combining datasets for
#      training (Step 1b)" section; left unset (the default), this track
#      is skipped entirely.
#   2. IMAGE below already points at the built .sif on this cluster
#      (matching the pinned requirements/train.txt environment; Step 5's
#      own deps -- pandas/numpy/matplotlib -- are already in that image
#      too, so no second image is needed for this single-job chain. The
#      GeoPackage street-map path, OSM_GEOPACKAGE_PATH, works here since
#      it's plain stdlib sqlite3; pyrosm/OSM_PBF_PATH does NOT, since
#      pyrosm isn't part of requirements/train.txt -- see README's "Notes
#      on the source notebooks" for why pyrosm was never straightforward
#      to install). Update IMAGE if that path changes, and point REPO_DIR
#      at this repository's root on the cluster's filesystem.
#   3. Point DATASETS_DIR (below) at wherever your dataset_labeled_w<W>.csv /
#      events_all_w<W>.csv files actually live on this cluster -- it can be
#      any absolute path, or a path relative to wherever you run `sbatch`
#      from (this project's own convention, see README, is a `datasets/`
#      folder as a sibling of the repo checkout; the default below instead
#      points straight at this cluster's real location). It's bound into
#      the container separately from REPO_DIR and deliberately NOT just
#      referenced as a relative "../..." path from inside /workflow:
#      Singularity's --bind only exposes the exact directory you name
#      (REPO_DIR -> /workflow here), not its parent, so a relative path
#      from /workflow would resolve outside anything actually mounted into
#      the container and fail, even though the identical relative path
#      works fine locally (not inside a container) or on a cluster whose
#      Singularity config happens to auto-bind the wider filesystem --
#      don't rely on that. DATASET_PATH/EVENTS_CSV_PATH/
#      CROSS_RUN_DATASET_PATHS/COMBINE_DATASET_PATHS/
#      COMBINE_CROSS_RUN_DATASET_PATHS below all use the in-container
#      /datasets/... path this mount creates, not the host path.
#   4. Adjust --nodelist (or drop it and let Slurm schedule any GPU node),
#      --cpus-per-task, and --time for your cluster -- this job runs
#      longer than train_model.sbatch alone (all five steps, not just
#      training), so a queue with a short per-job time limit may need a
#      raised --time here even if train_model.sbatch/evaluate_model.sbatch
#      individually fit under the default.
#   5. --env PYTHONNOUSERSITE=1 below stops your host ~/.local packages
#      (auto-mounted into the container along with the rest of $HOME, since
#      none of these templates pass --contain/--no-home) from shadowing the
#      container's own pinned dependencies via Python's user-site-packages
#      precedence -- see train_model.sbatch's header comment (point 4) for
#      the full explanation. Leave it in place unless you have a specific
#      reason to want host packages to take precedence.

#SBATCH --job-name fumd_run_pipeline
#SBATCH --output logs/run_pipeline_%j.out
#SBATCH --error logs/run_pipeline_%j.error
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpgpu01

REPO_DIR=.                              # this repository's root
DATASETS_DIR=/home/hpc/users/sonja.filiposka/resources/datasets    # this cluster's actual datasets location -- see point 3 above; adjust if yours lives elsewhere
IMAGE=/home/hpc/users/sonja.filiposka/image_jupiter_eosc.sif    # built from requirements/train.txt (see README)

# Optional: override this run's parameters instead of hand-editing the
# notebook (any parameter name from run_pipeline.ipynb's parameters cell
# works the same way). Dataset paths use the /datasets/... mount point
# this script binds DATASETS_DIR to (see point 3 above), e.g. for a
# DATASETS_DIR/1000_2/dataset_labeled_w3.csv file on the host:
# export DATASET_PATH=/datasets/1000_2/dataset_labeled_w3.csv
# export EVENTS_CSV_PATH=/datasets/1000_2/events_all_w3.csv
# export OUTPUT_DIR=pipeline_run_1000_2
# export EPOCHS=50
# export CROSS_RUN_DATASET_PATHS='["/datasets/900_1/dataset_labeled_w3.csv","/datasets/1000_1/dataset_labeled_w3.csv"]'
# export OSM_GEOPACKAGE_PATH=/absolute/path/on/cluster/roads.osm.geopackage.zip
#
# Leave COMBINE_DATASET_PATHS unset (the default) to run ONLY the main,
# single-dataset model above -- no combined-dataset track, same as a plain
# `jupyter execute run_pipeline.ipynb` with COMBINE_DATASET_PATHS=[]. Set
# it (same /datasets/... paths, see point 1 above) to also run that track
# in this same job.

NOTEBOOK=run_pipeline.ipynb

mkdir -p logs
singularity exec --nv --pwd /workflow \
    --env PYTHONNOUSERSITE=1 \
    --bind "${REPO_DIR}:/workflow" \
    --bind "${DATASETS_DIR}:/datasets" \
    "${IMAGE}" \
    jupyter execute "/workflow/${NOTEBOOK}"
