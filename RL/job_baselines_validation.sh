#!/bin/bash
#PBS -N base_val
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -J 20-22
#PBS -j oe

cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rl-robot

SEED=$PBS_ARRAY_INDEX
mkdir -p validation/logs

echo "seed=$SEED  host=$(hostname)  start=$(date)"

python -u train_phase1.py --seed "$SEED" --out-dir validation > "validation/logs/base_val_seed${SEED}.out" 2>&1

echo "end=$(date)"
