#!/bin/bash

#PBS -J 10-14
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=48:0:0
#PBS -N phase1_seed

cd $PBS_O_WORKDIR

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rl-robot

SEED=$PBS_ARRAY_INDEX
echo "Running Phase 1 training for seed=${SEED}"

python3 train_phase1.py --seed "${SEED}"
