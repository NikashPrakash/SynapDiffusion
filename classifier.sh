#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEGNeuralDecoding
#SBATCH --nodes=1
#SBATCH --time=15:00
#SBATCH --account=eecs448w24_class
#SBATCH --mem=2g

source env/bin/activate 
python3 classifier.py