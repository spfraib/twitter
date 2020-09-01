#!/usr/bin/env bash

DATA_PATH=$1
MODEL_NAME=$2
RUN_NAME=$3

MODEL_TYPE='bert'
PREPROCESSED='0'
mkdir results_simpletransformers_${RUN_NAME}

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_lost_job_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_lost_job_1mo.csv \
 --run_name ${RUN_NAME} \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${RUN_NAME}/lost_job_1mo
 echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed**************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_is_unemployed.csv \
 --eval_data_path ${DATA_PATH}/val_is_unemployed.csv \
 --run_name ${RUN_NAME} \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${RUN_NAME}/is_unemployed
echo '***********************DONE TRAINING ON LABEL is_unemployed******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search*****************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_job_search.csv \
 --eval_data_path ${DATA_PATH}/val_job_search.csv \
 --run_name ${RUN_NAME} \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${RUN_NAME}/job_search
echo '***********************DONE TRAINING ON LABEL job_search*********************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_is_hired_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_is_hired_1mo.csv \
 --run_name ${RUN_NAME} \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${RUN_NAME}/is_hired_1mo
echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer******************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_job_offer.csv \
 --eval_data_path ${DATA_PATH}/val_job_offer.csv \
 --run_name ${RUN_NAME} \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${RUN_NAME}/job_offer
echo '***********************DONE TRAINING ON LABEL job_offer**********************************************************'

