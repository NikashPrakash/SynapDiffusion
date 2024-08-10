#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEGBatchLoading
#SBATCH --account=eecs448w24_class
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=100g

source env/bin/activate 

if [ "$1" == "eeg" ]; then
   python eeg_load_batches.py
elif [ "$1" == "meg" ]; then
   python meg_load_batches.py
elif [ "$1" == "comb" ]; then
   python eeg_meg_comb_load_batches.py
else 
   echo "Not a valid data type"
fi