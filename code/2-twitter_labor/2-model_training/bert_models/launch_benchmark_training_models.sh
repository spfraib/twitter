#!/bin/bash

DATA_FOLDER=$1
COUNTRY_CODE=$2
MODEL_TYPE_1=$3
MODEL_TYPE_2=$4
MODE=$5

DATA_PATH=/scratch/mt4493/twitter_labor/twitter-labor-data/data/${DATA_FOLDER}/${COUNTRY_CODE}

#if [[ ${MODEL_TYPE_1} == *"/"* ]]; then
#  MODEL_TYPE_1_WITHOUT_SLASH=${MODEL_TYPE_1//[${SLASH}]/_}
#else
#  MODEL_TYPE_1_WITHOUT_SLASH=${MODEL_TYPE_1}
#fi
#
#if [[ ${MODEL_TYPE_2} == *"/"* ]]; then
#  MODEL_TYPE_2_WITHOUT_SLASH=${MODEL_TYPE_2//[${SLASH}]/_}
#else
#  MODEL_TYPE_2_WITHOUT_SLASH=${MODEL_TYPE_2}
#fi
#
#DATA_FOLDER_FINAL=${DATA_FOLDER}/benchmark_models_${MODEL_TYPE_1_WITHOUT_SLASH}_VS_${MODEL_TYPE_2_WITHOUT_SLASH}
#OUTPUT_PATH=${DATA_PATH}/
#rm -r ${OUTPUT_PATH}; mkdir ${OUTPUT_PATH}
#cp ${DATA_PATH}/train-test ${OUTPUT_PATH}

select_model_name () {
  MODEL_TYPE=$1
  if [ ${MODEL_TYPE} = "dccuchile/bert-base-spanish-wwm-cased" ] || [ ${MODEL_TYPE} = "DeepPavlov/bert-base-cased-conversational" ] || [ ${MODEL_TYPE} = "neuralmind/bert-base-portuguese-cased" ]; then
    MODEL_NAME="bert"
  elif [ ${MODEL_TYPE} = "vinai/bertweet-base" ]; then
    MODEL_NAME="bertweet"
  elif [ ${MODEL_TYPE} = "dlb/electra-base-portuguese-uncased-brwac" ]; then
    MODEL_NAME="electra"
  elif [ ${MODEL_TYPE} = "roberta-base" ]; then
    MODEL_NAME="roberta"
  elif [ ${MODEL_TYPE} = "xlm-roberta-base" ]; then
    MODEL_NAME="xlmroberta"
  fi
  echo ${MODEL_NAME}
  }

MODEL_NAME_1=$(select_model_name "${MODEL_TYPE_1}")
MODEL_NAME_2=$(select_model_name "${MODEL_TYPE_2}")

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/2-model_training/bert_models

if [ ${MODE} -eq 0 ]; then
   echo "Model comparison on same data";
   for i in {1..5}; do
    sbatch ${CODE_FOLDER}/train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_1} ${MODEL_TYPE_1} 0
    sbatch ${CODE_FOLDER}/train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_2} ${MODEL_TYPE_2} 0
   done
elif [ ${MODE} -eq 1 ]; then
   echo "Segment vs non segment"
   for i in {1..5}; do
    sbatch ${CODE_FOLDER}/train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_1} ${MODEL_TYPE_1} 0
    sbatch ${CODE_FOLDER}/train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_1} ${MODEL_TYPE_1} 1
   done
fi



#sbatch /scratch/mt4493/twitter_labor