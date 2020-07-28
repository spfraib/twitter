#!/usr/bin/env bash

DATA_FOLDER=$1
MODEL_TYPE=$2
MODEL_NAME=$3
PREPROCESSED=$4

if [[ ${MODEL_NAME} == *"/"* ]]
then
    MODEL_NAME_FOLDER=${MODEL_NAME##*/}
else
    MODEL_NAME_FOLDER=${MODEL_NAME}
fi


if [[ ${DATA_FOLDER} == *"may20"* ]] && [ ${PREPROCESSED} -eq 1 ]
then
  DATA_PATH=twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced/preprocessed_glove
elif [[ ${DATA_FOLDER} == *"may5"* ]] && [ ${PREPROCESSED} -eq 1 ]
then
  DATA_PATH=twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros/preprocessed_glove
elif [[ ${DATA_FOLDER} == *"may20"* ]] && [ ${PREPROCESSED} -ne 1 ]
then
  DATA_PATH=twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced
elif [[ ${DATA_FOLDER} == *"may5"* ]] && [ ${PREPROCESSED} -ne 1 ]
then
  DATA_PATH=twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros
elif [[ ${DATA_FOLDER} == *"jun3"* ]] && [ ${PREPROCESSED} -ne 1 ]
then
  DATA_PATH=twitter/data/${DATA_FOLDER}/data_binary_pos_neg_balanced
fi

mkdir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 twitter/code/8-training_binary/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_lost_job_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_lost_job_1mo.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_lost_job_1mo \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}
 echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed**************************************************'
python3 twitter/code/8-training_binary/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_is_unemployed.csv \
 --eval_data_path ${DATA_PATH}/val_is_unemployed.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_is_unemployed \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}
echo '***********************DONE TRAINING ON LABEL is_unemployed******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search*****************************************************'
python3 twitter/code/8-training_binary/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_job_search.csv \
 --eval_data_path ${DATA_PATH}/val_job_search.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_job_search \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}
echo '***********************DONE TRAINING ON LABEL job_search*********************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 twitter/code/8-training_binary/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_is_hired_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_is_hired_1mo.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_is_hired_1mo \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}
echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer******************************************************'
python3 twitter/code/8-training_binary/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_job_offer.csv \
 --eval_data_path ${DATA_PATH}/val_job_offer.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_job_offer \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}
echo '***********************DONE TRAINING ON LABEL job_offer**********************************************************'

