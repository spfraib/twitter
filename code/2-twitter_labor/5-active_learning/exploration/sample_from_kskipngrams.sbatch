#!/bin/bash

#SBATCH --job-name=sample_kskipngrams
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
##SBATCH --gres=gpu:1
#SBATCH --output=slurm_sample_kskipngrams_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

COUNTRY_CODE=$1
MODEL_FOLDER=$2

# array=0-49

module purge
#module load anaconda3/2020.02
#source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate
#cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/exploration
#LOG_FOLDER=log/${MODEL_FOLDER}
#mkdir -p ${LOG_FOLDER}
singularity exec --overlay /scratch/mt4493/twitter_labor/code/envs/singularity/kskipngrams/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/exploration/sample_from_kskipngrams.py \
	     --country_code ${COUNTRY_CODE} \
	     --model_folder ${MODEL_FOLDER}"


# sbatch --array=0-49 sample_from_kskipngrams.sbatch
