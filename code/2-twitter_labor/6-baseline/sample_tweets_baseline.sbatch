#!/bin/bash

#SBATCH --job-name=sample_tweets_baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_baseline-%j.out

COUNTRY_CODE=$1
INFERENCE_FOLDER=$2

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/6-baseline

python3 sample_tweets_baseline.py \
--country_code ${COUNTRY_CODE} \
--inference_folder ${INFERENCE_FOLDER}