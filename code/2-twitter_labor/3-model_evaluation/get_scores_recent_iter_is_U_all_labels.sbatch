#!/bin/bash

#SBATCH --job-name=get_scores
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_get_scores_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

SAM=1

if [ ${SAM} -eq 1 ]
then
  SCRATCH_PATH=/scratch/spf248/scratch_manu
elif [ ${SAM} -eq 0 ]
then
  SCRATCH_PATH=/scratch/mt4493
fi

module purge

singularity exec --overlay ${SCRATCH_PATH}/twitter_labor/code/envs/singularity/inference/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 ${SCRATCH_PATH}/twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/get_scores_recent_iter_is_U_all_labels.py"