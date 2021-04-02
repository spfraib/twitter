#!/bin/bash

#SBATCH --job-name=zip_folders
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_zip_folders-%j.out

cd /scratch/mt4493
zip -r womply_data.zip womply_data/

cd /scratch/mt4493/twitter_labor/twitter-labor-data/data
zip -r pretraining.zip pretraining/
echo 'Zipped pretraining '
zip -r tweets_2020.zip tweets_2020/
echo 'Zipped tweets_2020'
zip -r archive.zip archive/
echo 'Zipped archive'

