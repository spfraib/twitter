#!/bin/bash

#SBATCH --job-name=add_already_labelled
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_add_already_labelled-%j.out

COUNTRY_CODE=$1
INFERENCE_FOLDER=$2
NEW_ITERATION_FOLDER=$3

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning

python3 kskipngrams/add_already_labelled_tweets.py \
--country_code ${COUNTRY_CODE} \
--inference_folder ${INFERENCE_FOLDER} \
--new_iteration_folder "${NEW_ITERATION_FOLDER}"