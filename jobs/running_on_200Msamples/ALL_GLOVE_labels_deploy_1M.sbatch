#!/bin/bash

#SBATCH --job-name=GLOVE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.
#SBATCH --mail-type=END
#SBATCH --mail-user=nuclearr.d@gmail.com

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/2019.10
source /scratch/da2734/pyenv_dval_wb_twitter/py3.7/bin/activate
#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5
#module load miniconda


#/usr/bin/ssh -N -f -R $port:localhost:$port log-0
#/usr/bin/ssh -N -f -R $port:localhost:$port log-1


#unset XDG_RUNTIME_DIR
#if [ "$SLURM_JOBTMP" != "" ]; then
#    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
#fi


#source ~/miniconda3/bin/activate worldbank
#source activate worldbank
#conda activate worldbank

echo "pyenv activated"
#echo "shell" $0

export GENSIM_DATA_DIR=/scratch/da2734/twitter/code/11-baseline/logit_glove/downloaded

echo $GENSIM_DATA_DIR

#echo what

cd /scratch/da2734/twitter/code/10-deploy-100M-samples/

pwd

#jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
#python -u training_binary.py is_hired_1mo > /scratch/da2734/twitter/batch/train_bert_binary_separate_models_is_hired_1mo.log 2>&1
#srun time python -u 10.2-GLOVE-deploying-10M-randomONLY_1pct.py > /scratch/da2734/twitter/jobs/running_on_200Msamples/array_logs/ALL_10pct_GLOVE/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
srun time python -u 10.2-GLOVE-deploying-1M-randomONLY_1pct.py > /scratch/da2734/twitter/jobs/running_on_200Msamples/array_logs/ALL_1pct_GLOVE/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'running'

exit



