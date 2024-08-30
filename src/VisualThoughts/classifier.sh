#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=EEG_SVM_Classifier
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=8:00:00
#SBATCH --account=eecs448w24_class

source env/bin/activate
python3 classifier.py