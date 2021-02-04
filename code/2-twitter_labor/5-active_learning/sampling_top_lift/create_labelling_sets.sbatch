#!/bin/bash

#SBATCH --job-name=wordcount_random
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_wordcount_random_%j.out
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
ITER_NUMBER=$2
INFERENCE_FOLDER=$3

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/sampling_top_lift

python3 create_labelling_sets.py \
--country_code ${COUNTRY_CODE} \
--iter_number ${ITER_NUMBER} \
--inference_folder ${INFERENCE_FOLDER}
