#!/bin/bash

COUNTRY_CODE=$1
SURVEY_ID=$2

#SBATCH --job-name=operations
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation

python3 get_results_from_qualtrics.py \
--country_code ${COUNTRY_CODE} \
--surveyId ${SURVEY_ID}
