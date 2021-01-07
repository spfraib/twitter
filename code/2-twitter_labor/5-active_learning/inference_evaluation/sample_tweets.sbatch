#!/bin/bash

#SBATCH --job-name=sample_tweets
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_sample_tweets_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
MODEL_FOLDER=$2

module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/active_learning_env/bin/activate
echo "virtualenv activated"

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/inference_evaluation
cd ${CODE_FOLDER}

python3 sample_tweets.py \
--country_code ${COUNTRY_CODE} \
--model_folder ${MODEL_FOLDER}

