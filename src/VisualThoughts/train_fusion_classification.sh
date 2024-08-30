#!/bin/bash
#SBATCH --job-name=MEGNeuralDecoding
#SBATCH --account=eecs448w24_class
#SBATCH --partition=gpu
#SBATCH --time=40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=25g

source env/bin/activate

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export OMP_NUM_THREADS=1


srun torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip \
gen_trainer.py
