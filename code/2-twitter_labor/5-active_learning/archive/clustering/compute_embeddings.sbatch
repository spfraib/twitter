#!/bin/bash
#SBATCH --job-name=compute_embeddings
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_compute_embeddings_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
MODEL_FOLDER=$2

# Load packages and activate virtual environment
module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/active_learning_env/bin/activate
echo "virtualenv activated"

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/clustering
OUTPUT_FOLDER=/scratch/mt4493/twitter_labor/twitter-labor-data/data/tweet_embeddings/${COUNTRY_CODE}/${MODEL_FOLDER}

python3 ${CODE_FOLDER}/compute_embeddings.py \
--country_code ${COUNTRY_CODE} \
--model_folder ${MODEL_FOLDER} \
--output_folder ${OUTPUT_FOLDER}