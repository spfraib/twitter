#!/bin/bash

#SBATCH --job-name=get_friends_from_twitter_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_get_friends_from_twitter_api-%j.out

#normal parameters are 5 cpus, 60GB, 168h
COUNTRY_CODE=$1

module purge
module load anaconda3/2019.10

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/1-data_preparation/3-users

python3 3.6-get-friends-from-twitter-api.py \
--country_code ${COUNTRY_CODE}

