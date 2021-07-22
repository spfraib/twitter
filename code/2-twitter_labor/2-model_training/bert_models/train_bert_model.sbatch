#!/bin/bash
#SBATCH --job-name=training_berts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_training_berts_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

DATA_FOLDER=$1
COUNTRY_CODE=$2
MODEL_NAME=$3
MODEL_TYPE=$4
SEED=$5
INTRA_EPOCH_EVAL=$6
#SEGMENT=$5
#INTRA_EPOCH_EVAL=$5
TIMESTAMP=$(date +%s)
JOB_ID=${SLURM_JOB_ID}

SLASH=/
if [[ ${MODEL_TYPE} == *"/"* ]]; then
  MODEL_TYPE_WITHOUT_SLASH=${MODEL_TYPE//[${SLASH}]/-}
else
  MODEL_TYPE_WITHOUT_SLASH=${MODEL_TYPE}
fi

if [[ ${DATA_FOLDER} == *"/"* ]]; then
  DATA_FOLDER_WITHOUT_SLASH=${DATA_FOLDER//[${SLASH}]/_}
else
  DATA_FOLDER_WITHOUT_SLASH=${DATA_FOLDER}
fi

DATA_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/train_test/${COUNTRY_CODE}/${DATA_FOLDER}/train_test
#HOLDOUT_DATA_FOLDER=/scratch/mt4493/twitter_labor/twitter-labor-data/data/${DATA_FOLDER}

# Load packages and activate virtual environment
module purge
module load anaconda3/2020.02
source /scratch/mt4493/twitter_labor/code/envs/finetuning_bert_env/bin/activate
echo "virtualenv activated"

#create output folder
mkdir /scratch/mt4493/twitter_labor/trained_models/"${COUNTRY_CODE}"/"${MODEL_TYPE_WITHOUT_SLASH}"_"${DATA_FOLDER_WITHOUT_SLASH}"_"${JOB_ID}"_seed-"${SEED}"
OUTPUT_DIR=/scratch/mt4493/twitter_labor/trained_models/${COUNTRY_CODE}/${MODEL_TYPE_WITHOUT_SLASH}_${DATA_FOLDER_WITHOUT_SLASH}_${JOB_ID}_seed-${SEED}
echo "Output folder created: ${OUTPUT_DIR}"

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/2-model_training/bert_models

shopt -s extglob

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 train_bert.py \
  --train_data_path ${DATA_PATH}/train_lost_job_1mo.csv \
  --eval_data_path ${DATA_PATH}/val_lost_job_1mo.csv \
  --model_name ${MODEL_NAME} \
  --model_type ${MODEL_TYPE} \
  --num_train_epochs 10 \
  --output_dir ${OUTPUT_DIR}/lost_job_1mo \
  --slurm_job_timestamp ${TIMESTAMP} \
  --slurm_job_id ${JOB_ID} \
  --intra_epoch_evaluation ${INTRA_EPOCH_EVAL} \
  --seed ${SEED} \
  #--holdout_data_path ${DATA_PATH}/holdout_lost_job_1mo.csv \

rm -rf ${OUTPUT_DIR}/lost_job_1mo/models/!(best_model)
echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed***************************************************'
python3 train_bert.py \
  --train_data_path ${DATA_PATH}/train_is_unemployed.csv \
  --eval_data_path ${DATA_PATH}/val_is_unemployed.csv \
  --model_name ${MODEL_NAME} \
  --model_type ${MODEL_TYPE} \
  --num_train_epochs 10 \
  --output_dir ${OUTPUT_DIR}/is_unemployed \
  --slurm_job_timestamp ${TIMESTAMP} \
  --slurm_job_id ${JOB_ID} \
  --intra_epoch_evaluation ${INTRA_EPOCH_EVAL} \
  --seed ${SEED} \
  #--holdout_data_path ${DATA_PATH}/holdout_is_unemployed.csv \

rm -rf ${OUTPUT_DIR}/is_unemployed/models/!(best_model)

echo '***********************DONE TRAINING ON LABEL is_unemployed*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search***************************************************'
python3 train_bert.py \
  --train_data_path ${DATA_PATH}/train_job_search.csv \
  --eval_data_path ${DATA_PATH}/val_job_search.csv \
  --model_name ${MODEL_NAME} \
  --model_type ${MODEL_TYPE} \
  --num_train_epochs 10 \
  --output_dir ${OUTPUT_DIR}/job_search \
  --slurm_job_timestamp ${TIMESTAMP} \
  --slurm_job_id ${JOB_ID} \
  --intra_epoch_evaluation ${INTRA_EPOCH_EVAL} \
  --seed ${SEED} \
  #--holdout_data_path ${DATA_PATH}/holdout_job_search.csv \

rm -rf ${OUTPUT_DIR}/job_search/models/!(best_model)

echo '***********************DONE TRAINING ON LABEL job_search*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 train_bert.py \
  --train_data_path ${DATA_PATH}/train_is_hired_1mo.csv \
  --eval_data_path ${DATA_PATH}/val_is_hired_1mo.csv \
  --model_name ${MODEL_NAME} \
  --model_type ${MODEL_TYPE} \
  --num_train_epochs 10 \
  --output_dir ${OUTPUT_DIR}/is_hired_1mo \
  --slurm_job_timestamp ${TIMESTAMP} \
  --slurm_job_id ${JOB_ID} \
  --intra_epoch_evaluation ${INTRA_EPOCH_EVAL} \
  --seed ${SEED} \
  #--holdout_data_path ${DATA_PATH}/holdout_is_hired_1mo.csv \

rm -rf ${OUTPUT_DIR}/is_hired_1mo/models/!(best_model)

echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer***************************************************'
python3 train_bert.py \
  --train_data_path ${DATA_PATH}/train_job_offer.csv \
  --eval_data_path ${DATA_PATH}/val_job_offer.csv \
  --model_name ${MODEL_NAME} \
  --model_type ${MODEL_TYPE} \
  --num_train_epochs 10 \
  --output_dir ${OUTPUT_DIR}/job_offer \
  --slurm_job_timestamp ${TIMESTAMP} \
  --slurm_job_id ${JOB_ID} \
  --intra_epoch_evaluation ${INTRA_EPOCH_EVAL} \
  --seed ${SEED} \
  #--holdout_data_path ${DATA_PATH}/holdout_job_offer.csv \

rm -rf ${OUTPUT_DIR}/job_offer/models/!(best_model)

echo '***********************DONE TRAINING ON LABEL job_offer*******************************************************'

#srun time python -u 8.2-random-samples-UNDERsampled-separate-labels.py job_search > /scratch/da2734/twitter/running_on_200Msamples/array_logs/job_search/8.2-random-samples-UNDERsampled-separate-labels_${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
exit
