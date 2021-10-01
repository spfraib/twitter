#!/bin/bash

#SBATCH --job-name=get_filenames_from_tars
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00

COUNTRY_CODE=$1
TAR_TYPE=$2

module purge
module load anaconda3/2020.02

cd /scratch/spf248/twitter/code/twitter/code/5-demographics/get_pictures

python get_filenames_from_tars.py \
--country_code ${COUNTRY_CODE} \
--tar_type ${TAR_TYPE}



