#!/bin/bash
#SBATCH --comment="Ramda training job"
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:medium:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=ramda-paam
#SBATCH --output=logs/ramda_%j.out
#SBATCH --error=logs/ramda_%j.err

echo "Job started on $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

source /home/diego-soi/.mamba_init.sh
mamba activate gnndroid

echo "Current working directory: $SLURM_SUBMIT_DIR"

cd $SLURM_SUBMIT_DIR

python classification_pipeline.py

echo "Job finished"