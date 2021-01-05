#!/bin/bash

#SBATCH --job-name=create_tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_create_tasks-%j.out

COUNTRY_CODE=$1
N_WORKERS=$2
BLOCK_SIZE=$3
SURVEY_LINK=$4
VERSION_NB=$5
MODE=$6

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation/mturk

python3 create_tasks.py \
--country_code ${COUNTRY_CODE} \
--n_workers ${N_WORKERS} \
--block_size ${BLOCK_SIZE} \
--survey_link ${SURVEY_LINK} \
--version_number ${VERSION_NB} \
--mode ${MODE}

