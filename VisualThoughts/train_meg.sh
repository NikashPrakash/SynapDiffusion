#!/bin/bash
#SBATCH --job-name=MEG_Ray_Debug
#SBATCH --account=eecs448w24_class
#SBATCH --partition=gpu
#SBATCH --time=25:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=22g


source env/bin/activate
python meg_ray.py
