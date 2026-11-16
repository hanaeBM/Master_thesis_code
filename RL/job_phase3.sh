#!/bin/bash
#PBS -N llm_multi_test
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=48:00:00
#PBS -J 10-14
#PBS -j oe

cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rl-robot

export HF_HUB_OFFLINE=1

PROCESS=aggregator
SEED=$PBS_ARRAY_INDEX
mkdir -p test_run/logs

echo "seed=$SEED  process=$PROCESS  host=$(hostname)  start=$(date)"
nvidia-smi

python -u train_phase3.py --seed "$SEED" --process "$PROCESS" --model Qwen/Qwen3.8-27B-FP8 --hint --min-grasp-share 0.3 --grasp-from 0.0 --total-timesteps 100000 --eval-freq 5000 --snapshot-every 2000 --out-dir test_run > "test_run/logs/llm_multi_${PROCESS}_seed${SEED}.out" 2>&1

echo "end=$(date)"


