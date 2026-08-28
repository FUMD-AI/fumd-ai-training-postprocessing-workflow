#!/bin/bash
#
# Submits Step 3 (evaluate_model) to Slurm -- same pattern as
# train_model.sbatch (see that file's header comment). Kept on a GPU node
# by default since `model.predict(...)` benefits from one too, though
# Step 3 itself doesn't train anything.
#
# Before submitting: set this run's parameters (RUN_DIR,
# CROSS_RUN_DATASET_PATHS, ...) either by editing
# step_3_evaluate_model.ipynb's own "parameters" cell, or by exporting
# environment variables of the same name below the REPO_DIR/IMAGE lines --
# see train_model.sbatch's header comment for why this works. IMAGE below
# already points at the built .sif on this cluster -- update it (and
# REPO_DIR/--nodelist) as in train_model.sbatch if that changes.
#
# CROSS_RUN_DATASET_PATHS entries need to resolve inside the container, not
# just on the host -- DATASETS_DIR below is bound separately from REPO_DIR
# (Singularity's --bind only exposes the exact directory you name, not its
# parent, so a relative "../datasets/..." path from inside the container
# wouldn't resolve even though it works locally) and mounted at /datasets;
# point CROSS_RUN_DATASET_PATHS entries at /datasets/... accordingly, e.g.
# '["/datasets/900_1/dataset_labeled_w3.csv"]'. See run_pipeline.sbatch's
# header comment for the fuller explanation.
#
# --env PYTHONNOUSERSITE=1 below stops your host ~/.local packages (auto-
# mounted into the container along with the rest of $HOME) from shadowing
# the container's own pinned dependencies -- see train_model.sbatch's
# header comment (point 4) for the full explanation.
#
# --env XLA_FLAGS below (via CUDA_NVVM_DIR) points XLA at a real CUDA
# nvvm/libdevice + ptxas -- this image only ships CUDA runtime libraries,
# not the toolkit components XLA needs to JIT-compile Keras's own
# optimizer.apply_gradients step, so `model.evaluate`/`model.predict` here
# would otherwise crash the same way training does -- see
# train_model.sbatch's header comment (point 5) for the full explanation.

#SBATCH --job-name fumd_evaluate_model
#SBATCH --output logs/evaluate_model_%j.out
#SBATCH --error logs/evaluate_model_%j.error
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpgpu01

REPO_DIR=.
DATASETS_DIR=/home/hpc/users/sonja.filiposka/resources/datasets    # this cluster's actual datasets location -- see header comment above; adjust if yours lives elsewhere
CUDA_NVVM_DIR=/home/hpc/users/sonja.filiposka/resources/cuda_nvvm    # nvvm/libdevice + bin/ptxas -- see header comment above; adjust if yours lives elsewhere
IMAGE=/home/hpc/users/sonja.filiposka/image_jupiter_eosc.sif

# Optional: override this run's parameters (see train_model.sbatch):
# export RUN_DIR=/absolute/path/on/cluster/pipeline_run
# export CROSS_RUN_DATASET_PATHS='["/datasets/900_1/dataset_labeled_w3.csv"]'

NOTEBOOK=notebooks/step_3_evaluate_model.ipynb

mkdir -p logs
singularity exec --nv --pwd /workflow \
    --env PYTHONNOUSERSITE=1 \
    --env XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVVM_DIR}" \
    --bind "${REPO_DIR}:/workflow" \
    --bind "${DATASETS_DIR}:/datasets" \
    "${IMAGE}" \
    jupyter execute "/workflow/${NOTEBOOK}"
