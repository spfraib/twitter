#!/bin/bash

#SBATCH --job-name=Profiles
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --time=24:00:00

COUNTRY_CODE=$1

module purge
module load python3/intel/3.6.3
source /home/vp1274/venv/py3.6.3/bin/activate
cd /scratch/spf248/twitter/data/classification/US/scripts

python get_user_images.py \
--country_code ${COUNTRY_CODE}

