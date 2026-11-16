#!/bin/bash
#PBS -N llm_single
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -J 10-14
#PBS -j oe

cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rl-robot

export HF_HUB_OFFLINE=1

SEED=$PBS_ARRAY_INDEX
mkdir -p results logs

echo "seed=$SEED  host=$(hostname)  start=$(date)"
nvidia-smi

python -u train_phase2_llm.py --seed "$SEED" --model Qwen/Qwen3.8-27B-FP8 \
    --total-timesteps 2000000 \
    --plan-len 20 --revision-freq 100 \
    --max-tool-calls 3 --max-probe-episodes 10 \
    > "logs/llm_single_seed${SEED}.out" 2>&1

echo "end=$(date)"