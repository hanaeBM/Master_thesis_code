#!/bin/bash
#PBS -N llm_val
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=48:00:00
#PBS -J 20-22
#PBS -j oe

cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rl-robot

export HF_HUB_OFFLINE=1

SEED=$PBS_ARRAY_INDEX
mkdir -p validation/logs

echo "seed=$SEED  host=$(hostname)  start=$(date)"
nvidia-smi

python -u train_phase2.py --seed "$SEED" --model Qwen/Qwen3.8-27B-FP8 --hint --min-grasp-share 0.3 --out-dir validation > "validation/logs/llm_val_seed${SEED}.out" 2>&1

echo "end=$(date)"
