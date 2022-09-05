#!/bin/bash

#SBATCH --job-name=get_scores
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_get_scores_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu


module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation

python3 get_scores_from_best_models.py

exit