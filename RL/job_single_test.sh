#!/bin/bash
#PBS -N llm_test
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=02:00:00

#PBS -j oe

cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rl-robot

export HF_HUB_OFFLINE=1

SEED=20
mkdir -p test_run/logs

echo "seed=$SEED  host=$(hostname)  start=$(date)"
nvidia-smi

python -u train_phase2.py --seed "$SEED" --model Qwen/Qwen3.8-27B-FP8 --hint --min-grasp-share 0.3 --grasp-from 0.0 --total-timesteps 20000 --eval-freq 5000 --snapshot-every 2000 --out-dir test_run > "test_run/logs/llm_test_seed${SEED}.out" 2>&1

echo "end=$(date)"
