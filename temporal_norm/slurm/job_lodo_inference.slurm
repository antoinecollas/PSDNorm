#!/bin/bash

## JOB INFO
#SBATCH --job-name=lodo_mros_psdn
#SBATCH --output=slurm_logs/lodo_inference_mros_psdn.out
#SBATCH --error=slurm_logs/lodo_inference_mros_psdn.err

## NODE CONFIGURATION
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00

cd $WORK/test_time_temporal_normalization/temporal_norm/
## ENV ACTIVATION

module load pytorch-gpu/py3/2.6.0

## CODE EXECUTION
srun python run_LODO_inference.py --dataset MROS --percent 0.15 --norm PSDNorm