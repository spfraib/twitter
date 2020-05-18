#!/usr/bin/env bash

DATA_FOLDER=$1
MODEL_TYPE=$2
MODEL_NAME=$3

mkdir results
cd results
mkdir ${DATA_FOLDER}
cd ..

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 twitter/code/train_binary/train_binary.py --input_data_folder twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros \
 --results_folder results/${DATA_FOLDER} \
 --training_description bert_conversational_on_may5_data \
 --label lost_job_1mo \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20
echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed**************************************************'
python3 twitter/code/train_binary/train_binary.py --input_data_folder twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros \
 --results_folder results/${DATA_FOLDER} \
 --training_description bert_conversational_on_may5_data \
 --label is_unemployed \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20
echo '***********************DONE TRAINING ON LABEL is_unemployed******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search*****************************************************'
python3 twitter/code/train_binary/train_binary.py --input_data_folder twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros \
 --results_folder results/${DATA_FOLDER} \
 --training_description bert_conversational_on_may5_data \
 --label job_search \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20
echo '***********************DONE TRAINING ON LABEL job_search*********************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 twitter/code/train_binary/train_binary.py --input_data_folder twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros \
 --results_folder results/${DATA_FOLDER} \
 --training_description bert_conversational_on_may5_data \
 --label is_hired_1mo \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20
echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer******************************************************'
python3 twitter/code/train_binary/train_binary.py --input_data_folder twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros \
 --results_folder results/${DATA_FOLDER} \
 --training_description bert_conversational_on_may5_data \
 --label job_offer \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20
echo '***********************DONE TRAINING ON LABEL job_offer**********************************************************'



