#!/bin/bash

#SBATCH --job-name=identify_missing_pictures
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_missing_pictures_%j.out


COUNTRY_CODE=$1

module purge
module load anaconda3/2020.02

cd /scratch/spf248/twitter/code/twitter/code/5-demographics/get_pictures

python identify_missing_pictures.py \
--country_code ${COUNTRY_CODE}
