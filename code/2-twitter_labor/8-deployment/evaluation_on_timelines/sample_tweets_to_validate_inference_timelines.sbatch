#!/bin/bash

#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_sample_%j.out

COUNTRY_CODE=$1

module purge

singularity exec --overlay /scratch/mt4493/twitter_labor/code/envs/singularity/deployment/deployment.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/8-deployment/evaluation_on_timelines/sample_tweets_to_validate_inference_timelines.py --country_code ${COUNTRY_CODE}"