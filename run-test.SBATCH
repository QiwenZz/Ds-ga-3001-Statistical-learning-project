#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

python run.py
