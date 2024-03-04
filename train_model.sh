#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=MultimodalNeuralDecoding
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --account=eecs448w24_class
#SBATCH --partition=gpu
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=120gb
#SBATCH --gpus=6

module load python3.10-anaconda/2023.03
module load cuda/12.3.0
eval "$(conda shell.bash hook)"
conda activate MultimodalNeural

python3 training.py