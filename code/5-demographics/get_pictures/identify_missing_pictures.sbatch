#!/bin/bash

#SBATCH --job-name=identify_missing_pictures
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_missing_pictures_%j.out


COUNTRY_CODE=$1

module purge

singularity exec --overlay /scratch/spf248/twitter/code/singularity/get_pictures.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/spf248/twitter/code/twitter/code/5-demographics/get_pictures/identify_missing_pictures.py --country_code ${COUNTRY_CODE}"

