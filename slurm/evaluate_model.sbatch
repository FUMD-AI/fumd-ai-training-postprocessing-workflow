#!/bin/bash
#
# Submits Step 3 (evaluate_model) to Slurm -- same pattern as
# train_model.sbatch (see that file's header comment). Kept on a GPU node
# by default since `model.predict(...)` benefits from one too, though
# Step 3 itself doesn't train anything.
#
# Before submitting: edit step_3_evaluate_model.ipynb's own "parameters"
# cell (RUN_DIR, CROSS_RUN_DATASET_PATHS, ...) for this run, and update
# IMAGE/REPO_DIR/--nodelist as in train_model.sbatch.

#SBATCH --job-name fumd_evaluate_model
#SBATCH --output logs/evaluate_model_%j.out
#SBATCH --error logs/evaluate_model_%j.error
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpgpu01

REPO_DIR=.
IMAGE=image_cuda_jupyter.sif
NOTEBOOK=notebooks/step_3_evaluate_model.ipynb

mkdir -p logs
singularity exec --nv --pwd /workflow --bind "${REPO_DIR}:/workflow" "${IMAGE}" \
    jupyter execute "/workflow/${NOTEBOOK}"
