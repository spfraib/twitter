#!/usr/bin/env bash

DATA_FOLDER=$1
MODEL_TYPE=$2
MODEL_NAME=$3
PREPROCESSED=$4
NEW_TRAIN_TEST_SPLIT=$5
PARQUET_PATH='twitter/code/twitter_labor/model_training/saved_data'

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
fi

mkdir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}_new_data_combined_split

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_lost_job_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_lost_job_1mo.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_lost_job_1mo_new_data_combined_split \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}_new_data_combined_split \
 --add_new_labels True \
 --parquet_folder_path ${PARQUET_PATH} \
 --label_to_train_on 'lost_job_1mo' \
 --new_train_test_split ${NEW_TRAIN_TEST_SPLIT}
 echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed**************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_is_unemployed.csv \
 --eval_data_path ${DATA_PATH}/val_is_unemployed.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_is_unemployed_new_data_combined_split \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}_new_data_combined_split \
 --add_new_labels True \
 --parquet_folder_path ${PARQUET_PATH} \
 --label_to_train_on 'is_unemployed' \
 --new_train_test_split ${NEW_TRAIN_TEST_SPLIT}
echo '***********************DONE TRAINING ON LABEL is_unemployed******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search*****************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_job_search.csv \
 --eval_data_path ${DATA_PATH}/val_job_search.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_job_search_new_data_combined_split \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}_new_data_combined_split \
 --add_new_labels True \
 --parquet_folder_path ${PARQUET_PATH} \
 --label_to_train_on 'job_search' \
 --new_train_test_split ${NEW_TRAIN_TEST_SPLIT}
echo '***********************DONE TRAINING ON LABEL job_search*********************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_is_hired_1mo.csv \
 --eval_data_path ${DATA_PATH}/val_is_hired_1mo.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_is_hired_1mo_new_data_combined_split \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}_new_data_combined_split \
 --add_new_labels True \
 --parquet_folder_path ${PARQUET_PATH} \
 --label_to_train_on 'is_hired_1mo' \
 --new_train_test_split ${NEW_TRAIN_TEST_SPLIT}
echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer******************************************************'
python3 twitter/code/twitter_labor/model_training/simple_transformers/classification.py \
 --train_data_path ${DATA_PATH}/train_job_offer.csv \
 --eval_data_path ${DATA_PATH}/val_job_offer.csv \
 --preprocessed_input ${PREPROCESSED} \
 --run_name ${MODEL_NAME}_${DATA_FOLDER}_job_offer_new_data_combined_split \
 --model_type ${MODEL_TYPE} \
 --model_name ${MODEL_NAME} \
 --num_train_epochs 20 \
 --output_dir results_simpletransformers_${MODEL_NAME_FOLDER}_${DATA_FOLDER}_${PREPROCESSED}_new_data_combined_split \
 --add_new_labels True \
 --parquet_folder_path ${PARQUET_PATH} \
 --label_to_train_on 'job_offer' \
 --new_train_test_split ${NEW_TRAIN_TEST_SPLIT}
echo '***********************DONE TRAINING ON LABEL job_offer**********************************************************'
