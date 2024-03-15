#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEGNeuralDecoding
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --account=eecs448w24_class
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2g

source env/bin/activate 
python loadBatches.py