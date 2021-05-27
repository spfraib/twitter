#!/bin/bash

#SBATCH --job-name=undo_rejection
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_undo_rejection-%j.out

HITID=$1
WORKER_ID=$2

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation/mturk

echo "******** Accepting assignment for worker ${WORKER_ID} ********"
python3 undo_rejection.py \
--HITId ${HITId} \
--worker_id ${WORKER_ID}


