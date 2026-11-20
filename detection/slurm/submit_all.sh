#!/bin/bash
# Chain the whole detection run: prepare -> (YOLO, Mask R-CNN) in parallel. It changes to the kit folder itself:
#   bash detection/slurm/submit_all.sh                                    # visible masks
#   MASKS=amodal bash detection/slurm/submit_all.sh                       # amodal masks
set -euo pipefail
cd "$(dirname "$0")/.."
export MASKS=${MASKS:-visible}
P=$(sbatch --parsable --export=ALL slurm/00_prepare.sbatch)
Y=$(sbatch --parsable --export=ALL --dependency=afterok:$P slurm/10_yolo.sbatch)
M=$(sbatch --parsable --export=ALL --dependency=afterok:$P slurm/20_maskrcnn.sbatch)
echo "submitted: prepare $P, yolo $Y, maskrcnn $M (MASKS=$MASKS)"
