#!/bin/bash

#SBATCH --job-name=generate_user_image_map
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_generate_user_image_map_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu


module purge
module load anaconda3/2020.02

cd /scratch/spf248/twitter_deprecated/code/twitter/code/5-demographics/inference

python3 generate_user_image_map.py