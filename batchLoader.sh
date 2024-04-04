#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEGBatchLoading
#SBATCH --account=eecs448w24_class
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=6g

source env/bin/activate 

if [ "$1" == "eeg" ]; then
    python load_batches_eeg.py
else
    python load_batches_meg.py
fi