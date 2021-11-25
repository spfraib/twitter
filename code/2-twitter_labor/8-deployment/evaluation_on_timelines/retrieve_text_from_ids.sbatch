#!/bin/bash

#SBATCH --job-name=retrieve_text
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_retrieve_text_%j.out

COUNTRY_CODE=$1

JOB_ID=${SLURM_ARRAY_JOB_ID}

module purge

singularity exec --overlay /scratch/mt4493/twitter_labor/code/envs/singularity/inference/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/8-deployment/evaluation_on_timelines/retrieve_text_from_ids.py --country_code ${COUNTRY_CODE}"