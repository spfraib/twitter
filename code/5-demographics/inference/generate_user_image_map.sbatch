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
singularity exec --overlay /scratch/mt4493/just_another_day/singularity/tweepy.ext3:ro \
  /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
  /bin/bash -c "source /ext3/env.sh; python3 /scratch/mt4493/twitter_labor/code/twitter/code/5-demographics/inference/generate_user_image_map.py"