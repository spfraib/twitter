#!/bin/bash

#SBATCH --job-name=get_training_set_to_qualtrics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_get_training_set-%j.out

COUNTRY_CODE=$1
N_WORKERS=$2
BLOCK_SIZE=$3
VERSION_NB=$4
ITERATION_NB=$5
#MODE=$5

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation/qualtrics

#if [ ${MODE} -eq 0 ]
#then
python3 get_training_set_to_qualtrics_API_classification.py \
--country_code ${COUNTRY_CODE} \
--n_workers ${N_WORKERS} \
--block_size ${BLOCK_SIZE} \
--version_number ${VERSION_NB} \
--iteration_number ${ITERATION_NB}
#elif [ ${MODE} -eq 1 ]
#then
#  python3 get_training_set_to_qualtrics_API_NER.py \
#  --country_code ${COUNTRY_CODE} \
#  --n_workers ${N_WORKERS} \
#  --block_size ${BLOCK_SIZE} \
#  --version_number ${VERSION_NB}
#fi

