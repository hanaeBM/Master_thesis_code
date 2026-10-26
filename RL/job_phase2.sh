#!/bin/bash
#PBS -N llm_single
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=24:00:00
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

python train_phase2.py --seed "$SEED" > "logs/llm_single_seed${SEED}.out" 2>&1

echo "end=$(date)"