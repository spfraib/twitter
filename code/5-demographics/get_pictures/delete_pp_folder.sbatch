#!/bin/bash

#SBATCH --job-name=delete_pp_folder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_delete_pp_folder_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

rm -rf /scratch/spf248/twitter/data/demographics/profile_pictures/MX