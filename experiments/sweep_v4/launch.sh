#!/usr/bin/env bash
#SBATCH --job-name=sweep_v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --array=0-19

set -euo pipefail

ROOT="/data/connor/nsd-decoding"
cd $ROOT

EXP_NAME="sweep_v4"
EXP_DIR="experiments/${EXP_NAME}"

configs=(
    # 0: baseline (reference)
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes baseline"
    # --- vary n_components ---
    # 1-2
    "--n_components 128 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes nc128"
    "--n_components 512 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes nc512"
    # --- vary dropout ---
    # 3-4
    "--n_components 256 --depth 3 --dropout 0.3 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes do0.3"
    "--n_components 256 --depth 3 --dropout 0.7 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes do0.7"
    # --- vary depth ---
    # 5-6
    "--n_components 256 --depth 1 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes d1"
    "--n_components 256 --depth 6 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.01 --notes d6"
    # --- vary lr ---
    # 7-8
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 1e-4 --wd 0.01 --notes lr1e-4"
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 5e-4 --wd 0.01 --notes lr5e-4"
    # --- vary wd ---
    # 9-10
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.1 --notes wd0.1"
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.1 --lr 1e-3 --wd 0.001 --notes wd0.001"
    # --- vary drop_path ---
    # 11-12
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.0 --lr 1e-3 --wd 0.01 --notes dp0.0"
    "--n_components 256 --depth 3 --dropout 0.5 --drop_path 0.3 --lr 1e-3 --wd 0.01 --notes dp0.3"
    # --- promising combos: high reg + low lr ---
    # 13-15
    "--n_components 256 --depth 3 --dropout 0.7 --drop_path 0.2 --lr 5e-4 --wd 0.1 --notes highreg_lowlr"
    "--n_components 256 --depth 3 --dropout 0.7 --drop_path 0.1 --lr 1e-4 --wd 0.1 --notes highreg_vlowlr"
    "--n_components 512 --depth 3 --dropout 0.7 --drop_path 0.1 --lr 5e-4 --wd 0.01 --notes nc512_highreg"
    # --- shallow + more components ---
    # 16-17
    "--n_components 512 --depth 1 --dropout 0.5 --drop_path 0.0 --lr 1e-3 --wd 0.01 --notes nc512_d1"
    "--n_components 128 --depth 1 --dropout 0.5 --drop_path 0.0 --lr 1e-3 --wd 0.01 --notes nc128_d1"
    # --- deep + strong reg ---
    # 18-19
    "--n_components 256 --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4 --wd 0.01 --notes d6_highreg"
    "--n_components 512 --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4 --wd 0.01 --notes nc512_d6_highreg"
)
config=${configs[SLURM_ARRAY_TASK_ID]}

uv run python src/nsd_decoding/nsd_flat_cococlip_decoding_v4.py $config
