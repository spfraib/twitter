#!/bin/bash

#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_sample_%j.out

COUNTRY_CODE=$1

module purge
module load anaconda3/2020.02

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/8-deployment/evaluation_on_timelines

python3 sample_tweets_to_validate_inference_timelines.py --country_code ${COUNTRY_CODE}