#!/bin/bash
#SBATCH --job-name=clustering_inference_output
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=75GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_clustering_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

# Load packages and activate virtual environment
module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/active_learning_env/bin/activate
echo "virtualenv activated"

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/clustering

python3 ${CODE_FOLDER}/fast_clustering.py

