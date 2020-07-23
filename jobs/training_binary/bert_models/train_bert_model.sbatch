#!/bin/bash
#SBATCH --job-name=training_$3_$1_%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=manuel.tonneau@mailbox.org

DATA_FOLDER=$1
MODEL_NAME=$2
MODEL_TYPE=$3
TIMESTAMP=${SLURM_JOB_ID}
echo ${TIMESTAMP}

SLASH=/
if [[ ${MODEL_TYPE} == *"/"* ]]
then
  MODEL_TYPE_WITHOUT_SLASH=${MODEL_TYPE//[${SLASH}]/_}
else
  MODEL_TYPE_WITHOUT_SLASH=${MODEL_TYPE}
fi

if [[ ${DATA_FOLDER} == *"/"* ]]
then
    DATA_FOLDER_WITHOUT_SLASH=${DATA_FOLDER//[${SLASH}]/_}
else
    DATA_FOLDER_WITHOUT_SLASH=${DATA_FOLDER}
fi

DATA_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/${DATA_FOLDER}

# Load packages and activate virtual environment
module purge
module load anaconda3/2019.10
source /scratch/mt4493/twitter_labor/code/training_env/bin/activate
echo "virtualenv activated"

#create output folder
mkdir /scratch/mt4493/twitter_labor/trained_models/${MODEL_TYPE_WITHOUT_SLASH}_${DATA_FOLDER_WITHOUT_SLASH}_${TIMESTAMP}
OUTPUT_DIR=/scratch/mt4493/twitter_labor/trained_models/${MODEL_TYPE_WITHOUT_SLASH}_${DATA_FOLDER_WITHOUT_SLASH}_${TIMESTAMP}
echo "Output folder created: ${OUTPUT_DIR}"

cd /scratch/mt4493/twitter_labor/code/twitter/code/8-training_binary/bert_models

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 classification.py \
 --train_data_path ${DATA_PATH}/train_lost_job_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_lost_job_1mo.csv \
 --model_name ${MODEL_NAME} \
 --model_type ${MODEL_TYPE} \
 --num_train_epochs 20 \
 --output_dir ${OUTPUT_DIR} \
 --timestamp ${TIMESTAMP}
echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed***************************************************'
python3 classification.py \
 --train_data_path ${DATA_PATH}/train_is_unemployed.csv \
 --eval_data_path ${DATA_PATH}/val_is_unemployed.csv \
 --model_name ${MODEL_NAME} \
 --model_type ${MODEL_TYPE} \
 --num_train_epochs 20 \
 --output_dir ${OUTPUT_DIR} \
 --timestamp ${TIMESTAMP}
echo '***********************DONE TRAINING ON LABEL is_unemployed*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search***************************************************'
python3 classification.py \
 --train_data_path ${DATA_PATH}/train_job_search.csv \
 --eval_data_path ${DATA_PATH}/val_job_search.csv \
 --model_name ${MODEL_NAME} \
 --model_type ${MODEL_TYPE} \
 --num_train_epochs 20 \
 --output_dir ${OUTPUT_DIR} \
 --timestamp ${TIMESTAMP}
echo '***********************DONE TRAINING ON LABEL job_search*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 classification.py \
 --train_data_path ${DATA_PATH}/train_is_hired_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_is_hired_1mo.csv \
 --model_name ${MODEL_NAME} \
 --model_type ${MODEL_TYPE} \
 --num_train_epochs 20 \
 --output_dir ${OUTPUT_DIR} \
 --timestamp ${TIMESTAMP}
echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer***************************************************'
python3 classification.py \
 --train_data_path ${DATA_PATH}/train_job_offer.csv \
 --eval_data_path ${DATA_PATH}/val_job_offer.csv \
 --model_name ${MODEL_NAME} \
 --model_type ${MODEL_TYPE} \
 --num_train_epochs 20 \
 --output_dir ${OUTPUT_DIR} \
 --timestamp ${TIMESTAMP}
echo '***********************DONE TRAINING ON LABEL job_offer*******************************************************'

#srun time python -u 8.2-random-samples-UNDERsampled-separate-labels.py job_search > /scratch/da2734/twitter/running_on_200Msamples/array_logs/job_search/8.2-random-samples-UNDERsampled-separate-labels_${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
exit