#!/bin/bash
#BATCH --job-name=shhs_preprocessing
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH -p owners,mignot

# Load environment
module load py-h5py
module load py-tensorflow/1.8.0_py27
module load py-scipy
module load py-scikit-learn
module load py-scipystack
module load python

python ./stroke-deep-learning/generate_h5_files.py
