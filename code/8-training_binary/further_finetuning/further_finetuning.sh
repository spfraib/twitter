#!/usr/bin/env bash

MODEL_NAME_FOLDER='bert-base-cased-conversational'
DATA_FOLDER='may28_data'
PREPROCESSED='0'
PARQUET_PATH='twitter/code/8-training_binary/saved_data'
DATA_PATH='twitter/data/may20_9Klabels/data_binary_pos_neg_balanced'
MODEL_PATH='results_simpletransformers_bert-base-cased-conversational_may20_9Klabels_0/DeepPavlov'
mkdir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 twitter/code/8-training_binary/further_finetuning/further_finetuning.py \
 --parquet_folder_path ${PARQUET_PATH} \
 --eval_data_path ${DATA_PATH}/val_lost_job_1mo.csv \
 --run_name bert-base-cased-conversational_may28_lost_job_1mo \
 --model_type bert \
 --num_train_epochs 20 \
 --best_model_path ${MODEL_PATH}/bert-base-cased-conversational_may20_9Klabels_lost_job_1mo/models/checkpoint-360-epoch-2 \
 --label_to_train_on lost_job_1mo \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed***************************************************'
python3 twitter/code/8-training_binary/further_finetuning/further_finetuning.py \
 --parquet_folder_path ${PARQUET_PATH} \
 --eval_data_path ${DATA_PATH}/val_is_unemployed.csv \
 --run_name bert-base-cased-conversational_may28_is_unemployed \
 --model_type bert \
 --num_train_epochs 20 \
 --best_model_path ${MODEL_PATH}/bert-base-cased-conversational_may20_9Klabels_is_unemployed/models/checkpoint-366-epoch-1 \
 --label_to_train_on is_unemployed \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************DONE TRAINING ON LABEL is_unemployed*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search***************************************************'
python3 twitter/code/8-training_binary/further_finetuning/further_finetuning.py \
 --parquet_folder_path ${PARQUET_PATH} \
 --eval_data_path ${DATA_PATH}/val_job_search.csv \
 --run_name bert-base-cased-conversational_may28_job_search \
 --model_type bert \
 --num_train_epochs 20 \
 --best_model_path ${MODEL_PATH}/bert-base-cased-conversational_may20_9Klabels_job_search/models/checkpoint-219-epoch-1 \
 --label_to_train_on job_search \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************DONE TRAINING ON LABEL job_search*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 twitter/code/8-training_binary/further_finetuning/further_finetuning.py \
 --parquet_folder_path ${PARQUET_PATH} \
 --eval_data_path ${DATA_PATH}/val_is_hired_1mo.csv \
 --run_name bert-base-cased-conversational_may28_is_hired_1mo \
 --model_type bert \
 --num_train_epochs 20 \
 --best_model_path ${MODEL_PATH}/bert-base-cased-conversational_may20_9Klabels_is_hired_1mo/models/checkpoint-298-epoch-2 \
 --label_to_train_on is_hired_1mo \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer***************************************************'
python3 twitter/code/8-training_binary/further_finetuning/further_finetuning.py \
 --parquet_folder_path ${PARQUET_PATH} \
 --eval_data_path ${DATA_PATH}/val_job_offer.csv \
 --run_name bert-base-cased-conversational_may28_job_offer \
 --model_type bert \
 --num_train_epochs 20 \
 --best_model_path ${MODEL_PATH}/bert-base-cased-conversational_may20_9Klabels_job_offer/models/checkpoint-1041-epoch-3 \
 --label_to_train_on job_offer \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************DONE TRAINING ON LABEL job_offer*******************************************************'