#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:default         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --partition=ampere
#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#!SBATCH --exclusive           # Exclusive mode
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    # In this case, 1 torchrun per node
#SBATCH --cpus-per-task=4

## cd {{ cwd }}
## 
## echo "sbatch {{ path_output_slurm }}"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun                        \
torchrun                    \
--nnodes 2                  \
--nproc_per_node 1          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_endpoint $head_node_ip:29500 \
train.py
