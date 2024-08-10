#!/bin/bash
#SBATCH --job-name=MEGNeuralDecoding
#SBATCH --account=eecs448w24_class
#SBATCH --partition=gpu
#SBATCH --time=60:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=16g

source env/bin/activate
python eeg_trainer.py
