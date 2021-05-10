#!/bin/bash

#SBATCH --job-name=diversity_numba
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_evaluation_diversity_numba_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

METHOD=$1

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/diversity

python3 compute_distance_numba.py \
--method ${METHOD}

exit