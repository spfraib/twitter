#!/bin/bash

#SBATCH --job-name=active_learning
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_active_learning_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

INFERENCE_FOLDER=$1

module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning

python3 select_tweets_to_label.py \
--inference_output_folder /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output \
XXX
