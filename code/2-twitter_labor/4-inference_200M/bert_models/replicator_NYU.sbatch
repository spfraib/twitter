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
module load cuda/11.0.194

source /scratch/mt4493/twitter_labor/code/envs/inference_2021_env/bin/activate

echo "pyenv activated"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_modelsn

pwd

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
      echo SLURM_ARRAY_TASK_ID=test
fi
echo 'running inference..'
srun time python -u replicator_NYU.py > dhaval_gpu_test.log 2>&1
echo 'done'

if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

singularity exec $nv \
            --overlay /scratch/mt4493/jupyter_old/manu.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            /bin/bash -c "
#source /ext3/env.sh
source /scratch/mt4493/twitter_labor/code/envs/inference_2021_env/bin/activate
# jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
echo 'running inference..'
srun time python -u replicator_NYU.py > dhaval_gpu_test.log 2>&1
echo 'done'
"



exit



