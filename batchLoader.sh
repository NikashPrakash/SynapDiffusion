#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEGBatchLoading
#SBATCH --account=eecs448w24_class
#SBATCH --nodes=1
#SBATCH --time=15:00
#SBATCH --mem=6g

source env/bin/activate 

if [ "$1" == "eeg" ]; then
   python eeg_load_batches.py
else
   python meg_load_batches.py
fi