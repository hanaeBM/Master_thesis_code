#!/bin/bash
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:10:00
#PBS -N test_pcgym

cd $PBS_O_WORKDIR
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate rl-distillation



