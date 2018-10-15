#!/bin/bash
#SBATCH --job-name=screening
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -p mignot,gpu,owners
#SBATCH --gres gpu:1
# Load environment

module load devel
module load python
module load py-h5py 
module load py-scikit-learn
module load py-tensorflow/1.8.0_py27

python ./stroke-deep-learning/train.py $1


