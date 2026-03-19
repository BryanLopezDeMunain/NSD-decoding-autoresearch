#!/usr/bin/env bash
#SBATCH --job-name=sweep_all_v1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --qos=high
#SBATCH --array=0

set -euo pipefail

ROOT="/data/connor/nsd-decoding"
cd $ROOT

EXP_DIR="experiments/sweep_all_v1"

configs=(
    "beta|ood"
    "beta|subj01"
    "full_ts|ood"
    "full_ts|subj01"
)
config=${configs[SLURM_ARRAY_TASK_ID]}

space=$(echo $config | cut -d '|' -f 1)
subset=$(echo $config | cut -d '|' -f 2)

if [[ $space == "full_ts" ]]; then
    script="nsd_flat_cococlip_full_ts_decoding.py"
else
    script="nsd_flat_cococlip_decoding.py"
fi

if [[ $subset == "ood" ]]; then
    args="--ood"
else
    args="--subs 0"
fi

lrs=( 3e-4 1e-3 3e-3 )

for lr in "${lrs[@]}"; do
    uv run python src/nsd_decoding/$script \
        $args \
        --lr $lr \
        --notes "${space}__${subset}__lr${lr}"
done
