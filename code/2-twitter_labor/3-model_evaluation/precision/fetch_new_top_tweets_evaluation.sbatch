#!/bin/bash

#SBATCH --job-name=fetch_new_top_tweets_evaluation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_fetch_new_top_tweets_evaluation-%j.out

COUNTRY_CODE=$1

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/precision

python3 fetch_new_top_tweets_evaluation.py \
--country_code ${COUNTRY_CODE}