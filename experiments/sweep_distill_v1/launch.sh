#!/usr/bin/env bash
#SBATCH --job-name=sweep_distill_v1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --qos=low
#SBATCH --account=training
#SBATCH --array=0-3

set -euo pipefail

ROOT="/data/connor/nsd-decoding"
cd $ROOT

SCRIPT="src/nsd_decoding/nsd_flat_cococlip_decoding_distill_v1.py"
EXP_DIR="experiments/sweep_distill_v1"

# Subject groups
sub_groups=(
    "0"
    "1"
    "2"
    "0,1,2"
)
subs=${sub_groups[SLURM_ARRAY_TASK_ID]}

# lr x wd grid (7 x 3 = 21 configs)
lrs=( 5e-5 1e-4 2e-4 3e-4 5e-4 1e-3 2e-3 )
wds=( 0.001 0.01 0.1 )

for lr in "${lrs[@]}"; do
    for wd in "${wds[@]}"; do
        uv run python $SCRIPT --subs $subs --lr $lr --wd $wd \
            --notes "s${subs}_lr${lr}_wd${wd}"
    done
done
