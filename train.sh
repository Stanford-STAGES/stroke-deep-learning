#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p mignot,gpu,owners
#SBATCH --gres gpu:1
#SBATCH -C GPU_MEM:16GB

# Load environment

module load py-h5py
module load py-tensorflow/1.8.0_py27
module load py-scipy
module load py-scikit-learn
module load py-scipystack
module load python


python ./stroke-deep-learning/train.py $1


