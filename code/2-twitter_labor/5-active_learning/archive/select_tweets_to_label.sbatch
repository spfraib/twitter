#!/bin/bash

#SBATCH --job-name=active_learning
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --output=slurm_active_learning_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

INFERENCE_FOLDER=$1
DATA_FOLDER=$2

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning

python3 select_tweets_to_label.py \
--inference_output_folder /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output \
--tweets_to_label_output_path /scratch/mt4493/twitter_labor/twitter-labor-data/data/${DATA_FOLDER}/tweets_to_label \
--train_data_folder /scratch/mt4493/twitter_labor/twitter-labor-data/data/${DATA_FOLDER} \
--exploit_method 'top_tweets' \
--nb_tweets_exploit 100 \
--nb_top_lift_kw 10 \
--nb_tweets_per_kw_mlm 10 \
--nb_kw_per_tweet_mlm 3 \
--nb_tweets_per_kw_final 30 \
--nb_top_kskipngrams 5 \
--nb_tweets_per_kskipngrams 30 \
--k_skipgram 2 \
--n_skipgram 4

