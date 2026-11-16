#!/bin/bash
#PBS -N test_stack
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=48:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rl-robot

SEED=10
mkdir -p logs

echo "Running test stack training for seed=${SEED}  host=$(hostname)  start=$(date)"

python -u test_stack.py --seed "$SEED" --total-timesteps 1000000 --eval-freq 5000 > "logs/test_stack_seed${SEED}.out" 2>&1

echo "end=$(date)"