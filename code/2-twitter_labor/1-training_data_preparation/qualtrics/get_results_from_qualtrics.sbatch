#!/bin/bash

#SBATCH --job-name=get_results_from_qualtrics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_get_results-%j.out

COUNTRY_CODE=$1
SURVEY_ID=$2
REJECT_BOTS=$3
HITID=$4
ITERATION_NB=$5

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation

echo "******** Getting results for survey ${SURVEY_ID} ********"
python3 qualtrics/get_results_from_qualtrics.py \
--country_code ${COUNTRY_CODE} \
--surveyId ${SURVEY_ID} \
--reject_bots ${REJECT_BOTS} \
--HITId ${HITID} \
--iteration_number ${ITERATION_NB}


echo '******** Merging results from Qualtrics ********'
python3 qualtrics/merge-all-qualtrics-results.py \
--country_code ${COUNTRY_CODE} \
--iteration_number ${ITERATION_NB}

echo '******** Assigning qualification to past respondents ********'
python3 mturk/assign_qualification_to_past_respondents_bots.py \
--country_code ${COUNTRY_CODE} \
--iteration_number ${ITERATION_NB}
