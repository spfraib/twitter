#!/bin/bash

#SBATCH --job-name=grant_permissions
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_grant_permission_%j.out

setfacl -R -m u:mt4493:rwx /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed
setfacl -R -m u:mt4493:rwx /scratch/spf248/twitter/data/user_timeline
setfacl -R -m u:mt4493:rwx /scratch/spf248/twitter/data
setfacl -R -m u:mt4493:rwx /scratch/spf248/twitter