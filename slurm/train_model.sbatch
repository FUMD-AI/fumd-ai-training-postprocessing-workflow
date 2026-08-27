#!/bin/bash
#
# Submits Step 2 (train_model) to Slurm on a GPU node, via the same
# `singularity exec ... jupyter execute <notebook>.ipynb` invocation this
# project uses for every GPU step -- no papermill/parameter-injection tool
# involved (see src/fumd_training_workflow/notebook_runner.py's docstring).
#
# Before submitting:
#   1. Edit step_2_train_model.ipynb's own "parameters" cell (INPUT_DIR,
#      OUTPUT_DIR, EPOCHS, ...) for this run -- `jupyter execute` has no
#      way to pass parameters in from outside the notebook.
#   2. Point IMAGE below at your built .sif (matching the pinned
#      requirements/train.txt environment) and REPO_DIR at this
#      repository's root on the cluster's filesystem.
#   3. Adjust --nodelist (or drop it and let Slurm schedule any GPU node)
#      and --cpus-per-task for your cluster.

#SBATCH --job-name fumd_train_model
#SBATCH --output logs/train_model_%j.out
#SBATCH --error logs/train_model_%j.error
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpgpu01

REPO_DIR=.                              # this repository's root
IMAGE=image_cuda_jupyter.sif            # built from requirements/train.txt (see README)
NOTEBOOK=notebooks/step_2_train_model.ipynb

mkdir -p logs
singularity exec --nv --pwd /workflow --bind "${REPO_DIR}:/workflow" "${IMAGE}" \
    jupyter execute "/workflow/${NOTEBOOK}"
