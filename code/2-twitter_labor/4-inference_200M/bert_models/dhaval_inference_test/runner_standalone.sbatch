#!/bin/bash

#SBATCH --job-name=dval_standalone
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com
#SBATCH --output=slurm_inference_berts_%j.out



module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_2021_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/dhaval_inference_test
pwd

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
      echo SLURM_ARRAY_TASK_ID=test
fi
echo 'running inference..'
#srun time python -u inference_ONNX_bert_100M_random_dhaval_optimized_batch_compared_torch_and_onnx_NYU_standalone.py > dhaval_inference_ONNX_bert_100M_random_dhaval_optimized_batch_compared_torch_and_onnx_NYU_standalone.log 2>&1
srun time python -u onnx_baselines.py > dhaval_onnx_baselines.py.log 2>&1
echo 'done'

exit



