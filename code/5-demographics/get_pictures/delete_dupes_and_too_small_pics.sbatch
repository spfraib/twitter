#!/bin/bash

#SBATCH --job-name=get_filenames_from_tars
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1

module purge
module load anaconda3/2020.02

cd /scratch/spf248/twitter/code/twitter/code/5-demographics/get_pictures

python delete_dupes_and_too_small_pics.py \
--country_code ${COUNTRY_CODE}