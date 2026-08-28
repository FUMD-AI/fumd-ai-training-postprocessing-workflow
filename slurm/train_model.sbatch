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
#   2. IMAGE below already points at the built .sif on this cluster
#      (matching the pinned requirements/train.txt environment); update it
#      if that path changes. Point REPO_DIR at this repository's root on
#      the cluster's filesystem.
#   3. Adjust --nodelist (or drop it and let Slurm schedule any GPU node)
#      and --cpus-per-task for your cluster.
#   4. This project's Singularity images are run without --contain, so by
#      default Singularity auto-mounts your host $HOME into the container --
#      and Python's user-site-packages (~/.local/lib/pythonX.Y/site-packages)
#      take precedence over the container's own pinned packages, silently
#      shadowing them if you happen to have anything installed there on the
#      host. The `--env PYTHONNOUSERSITE=1` below disables that lookup, so
#      the container always uses its own pinned dependency versions
#      regardless of what's in your host ~/.local. Leave it in place unless
#      you have a specific reason to want host packages to take precedence.
#   5. This image is also missing the CUDA toolkit's nvvm/libdevice and
#      ptxas components -- only the CUDA *runtime* libraries (cuDNN/
#      cuBLAS/cuFFT) are present -- which XLA needs to JIT-compile certain
#      GPU kernels, notably Keras's own optimizer.apply_gradients step,
#      which is unconditionally XLA-compiled regardless of anything in
#      this project's own notebook code. Without this, training crashes on
#      the very first step: "libdevice not found at ./libdevice.10.bc"
#      if only libdevice is missing, or "Failed to launch ptxas" if ptxas
#      is missing too (as in this image). CUDA_NVVM_DIR below points
#      XLA_FLAGS at a directory containing <dir>/nvvm/libdevice/
#      libdevice.10.bc and <dir>/bin/ptxas -- these were extracted once
#      from another CUDA-toolkit-containing image already present on this
#      cluster and copied to a location this project owns, independent of
#      that other image (the exact source CUDA version doesn't need to
#      match this image's own CUDA 11.8 runtime: ptxas is backward-
#      compatible with older PTX, and libdevice.10.bc is stable IR
#      bitcode -- confirmed working here using CUDA 12.6's copies of
#      both). Since $HOME is auto-mounted (point 4 above), CUDA_NVVM_DIR
#      just needs to be a real path under $HOME -- no extra --bind
#      required. The real fix is building image_jupiter_eosc.sif with the
#      CUDA toolkit's nvcc/nvvm component included so this workaround
#      isn't needed -- see requirements/train.txt's own note on this.

#SBATCH --job-name fumd_train_model
#SBATCH --output logs/train_model_%j.out
#SBATCH --error logs/train_model_%j.error
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpgpu01

REPO_DIR=.                              # this repository's root
IMAGE=/home/hpc/users/sonja.filiposka/image_jupiter_eosc.sif            # built from requirements/train.txt (see README)
CUDA_NVVM_DIR=/home/hpc/users/sonja.filiposka/resources/cuda_nvvm    # nvvm/libdevice + bin/ptxas -- see point 5 above; adjust if yours lives elsewhere

# Optional: override this run's parameters instead of hand-editing the
# notebook (any parameter name from its parameters cell works the same way):
# export OUTPUT_DIR=/absolute/path/on/cluster/pipeline_run
# export DATASET_PATH=/absolute/path/on/cluster/dataset_labeled_w3.csv

NOTEBOOK=notebooks/step_2_train_model.ipynb

mkdir -p logs
singularity exec --nv --pwd /workflow \
    --env PYTHONNOUSERSITE=1 \
    --env XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVVM_DIR}" \
    --bind "${REPO_DIR}:/workflow" "${IMAGE}" \
    jupyter execute "/workflow/${NOTEBOOK}"
