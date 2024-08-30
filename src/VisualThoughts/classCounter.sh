#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEGNeuralDecoding
#SBATCH --nodes=1
#SBATCH --time=3:00
#SBATCH --account=eecs448w24_class
#SBATCH --partition=gpu
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=500m
#SBATCH --gpus=1
#SBATCH --gpu_cmode=shared


source env/bin/activate
python3 classVis.py