#!/bin/bash

DATA_FOLDER=$1
MODEL_NAME=$2
MODEL_TYPE=$3

if [[ ${MODEL_TYPE} == *"/"* ]]
then
    MODEL_SUBFOLDER=${MODEL_TYPE##*/}
else
    MODEL_SUBFOLDER=${MODEL_TYPE}

SBATCH --job-name=training_bert_models
SBATCH --nodes=1
SBATCH --cpus-per-task=1
SBATCH --mem=20GB
SBATCH --time=24:00:00
SBATCH --gres=gpu:1

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.
SBATCH --mail-type=END
SBATCH --mail-user=manuel.tonneau@mailbox.org

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/2019.10
source /scratch/mt4493/twitter_labor/code/training_env/bin/activate
#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5
#module load miniconda


#/usr/bin/ssh -N -f -R $port:localhost:$port log-0
#/usr/bin/ssh -N -f -R $port:localhost:$port log-1


#unset XDG_RUNTIME_DIR
#if [ "$SLURM_JOBTMP" != "" ]; then
#    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
#fi

echo "pyenv activated"
#echo "shell" $0

cd /scratch/mt4493/twitter/code/8-training_binary/bert_models
pwd

#jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
#python -u training_binary_dhaval.py is_hired_1mo > /scratch/da2734/twitter/jobs/training_binary/terminal_logs/train_bert_binary_separate_models_is_hired_1mo.log 2>&1
python3 classification.py
#srun time python -u 8.2-random-samples-UNDERsampled-separate-labels.py job_search > /scratch/da2734/twitter/running_on_200Msamples/array_logs/job_search/8.2-random-samples-UNDERsampled-separate-labels_${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'running'
exit