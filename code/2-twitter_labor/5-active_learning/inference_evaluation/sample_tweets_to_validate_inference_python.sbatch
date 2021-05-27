#!/bin/bash

#SBATCH --job-name=sample_tweets_to_validate_inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_sample_tweets-%j.out

COUNTRY_CODE=$1
MODEL_FOLDER=$2

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning

python3 inference_evaluation/sample_tweets_to_validate_inference_python.py \
--country_code ${COUNTRY_CODE} \
--model_folder ${MODEL_FOLDER}