#!/bin/bash

#SBATCH --job-name=get_images
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --output=outfiles/slurm_get_images_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

MODE=$1


module purge

singularity exec \
        --bind /home/mt4493/resolv.conf:/etc/resolv.conf:ro \
        --overlay /scratch/mt4493/just_another_day/singularity/tweepy.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python /scratch/mt4493/twitter_labor/code/twitter/code/5-demographics/get_pictures/get_user_images.py --mode ${MODE}"


