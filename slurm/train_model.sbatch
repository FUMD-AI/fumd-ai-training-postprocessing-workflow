#!/bin/bash
#
# Submits Step 2 (train_model) to Slurm on a GPU node, via the same
# `singularity exec ... jupyter execute <notebook>.ipynb` invocation this
# project uses for every GPU step -- no papermill/parameter-injection tool
# involved (see src/fumd_training_workflow/notebook_runner.py's docstring).
#
# Before submitting:
#   1. Set this run's parameters (INPUT_DIR, OUTPUT_DIR, EPOCHS, ...) either
#      by editing step_2_train_model.ipynb's own "parameters" cell directly,
#      or by exporting environment variables of the same name below --
#      singularity exec (no --cleanenv here) passes the submitting shell's
#      environment through into the container, and each notebook's
#      env-override cell (right after its parameters cell) reads them.
#      `jupyter execute` itself has no -p/parameter flag (unlike papermill);
#      see src/fumd_training_workflow/notebook_runner.py's env_override()
#      docstring for exact value-parsing rules.
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

# Optional: override this run's parameters instead of hand-editing the
# notebook (any parameter name from its parameters cell works the same way):
# export OUTPUT_DIR=/absolute/path/on/cluster/pipeline_run
# export DATASET_PATH=/absolute/path/on/cluster/dataset_labeled_w3.csv

NOTEBOOK=notebooks/step_2_train_model.ipynb

mkdir -p logs
singularity exec --nv --pwd /workflow --bind "${REPO_DIR}:/workflow" "${IMAGE}" \
    jupyter execute "/workflow/${NOTEBOOK}"
