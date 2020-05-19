#!/usr/bin/env bash

DATA_FOLDER=$1
TWITTER_REPO_PATH=$2

DATA_PATH=${TWITTER_REPO_PATH}/data/${DATA_FOLDER}/data_binary_pos_neg_balanced_removed_allzeros
mkdir ${DATA_PATH}/preprocessed_glove

ls ${DATA_PATH} -I "label_*" | xargs -n 1 -P 2 -I {} python preprocessing.py --input_file_path ${DATA_PATH}/{} --output_folder ${DATA_PATH}/preprocessed_glove
