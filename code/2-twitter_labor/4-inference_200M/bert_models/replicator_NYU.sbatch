#!/bin/bash

#SBATCH --job-name=ONNX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu
#SBATCH --output=slurm_inference_berts_%j.out

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_2021_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models

pwd

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
      echo SLURM_ARRAY_TASK_ID=test
fi
echo 'running inference..'
srun time python -u replicator_NYU.py > dhaval_gpu_test.log 2>&1
echo 'done'

exit



