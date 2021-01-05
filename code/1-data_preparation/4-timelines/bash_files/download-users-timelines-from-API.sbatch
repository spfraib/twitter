#!/bin/bash

#SBATCH --job-name=download_timelines
#SBATCH --cpus-per-task=5
#SBATCH --mem=60GB
#SBATCH --time=168:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_download_timelines-%j.out

COUNTRY_CODE=$1
MODE=$2
THIS_BATCH=$3

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/1-data_preparation/4-timelines

echo ${MODE}

if [ ${MODE} = "get" ]
then
  python3 download-users-timelines-from-API.py \
  --country_code ${COUNTRY_CODE} \
  --cutoff 500 \
  --get 1 \
  --update 0

elif [ ${MODE} = "update" ]
then
  python3 4.6-update-users-timeline-from-Twitter-API_new.py \
  --country_code ${COUNTRY_CODE} \
  --cutoff 1000 \
  --this_batch ${THIS_BATCH}
else
  echo 'Mode not recognized'
fi

