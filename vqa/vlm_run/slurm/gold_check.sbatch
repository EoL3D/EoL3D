#!/bin/bash
# Gold self-check of the pack on the cluster: no model, no GPU; every task must score 1.000.
#   sbatch --export=ALL,PACK=$DATA/vlm/vlmpack_2026-11-05 gold_check.sbatch
#SBATCH --job-name=vlm_gold
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
# EDIT: your CPU partition if the default is not one, e.g.
##SBATCH --partition=cpu
set -euo pipefail

# EDIT: load Python the way your cluster provides it, then the environment, for example
# module purge
# module load Python/3.12
source "${VENV:-$SCRATCH/venvs/vlmpack}/bin/activate"

# the pack's export units (manifest.json "units": the camera views with records, then multi;
# an empty unit is not in the pack); submit_matrix.sh passes VIEW_LIST and a matching --array
: "${PACK:?set PACK}"
read -r -a VIEWS <<< "${VIEW_LIST:-$(bash "$PACK/tools/vlm_run/pack_units.sh" "$PACK")}"
if (( SLURM_ARRAY_TASK_ID >= ${#VIEWS[@]} )); then
  echo "[${SLURM_JOB_NAME:-vlm}] array task $SLURM_ARRAY_TASK_ID: the pack has ${#VIEWS[@]} units (${VIEWS[*]}), nothing to run"
  exit 0
fi
export PACK="${PACK:?set PACK}"
export VIEW="${VIEWS[$SLURM_ARRAY_TASK_ID]}"
export BACKEND=gold
export MODEL_TAG=gold
export VARIANT="${VARIANT:-image}"
export SUBSET="${SUBSET:-eval}"
bash "$PACK/tools/vlm_run/run_eval.sh"
