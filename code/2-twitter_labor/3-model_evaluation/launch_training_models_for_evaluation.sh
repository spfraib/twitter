#!/bin/bash

DATA_FOLDER=$1
COUNTRY_CODE=$2
MODEL_TYPE_1=$3

SAM=0

if [ ${SAM} -eq 1 ]
then
  SCRATCH_PATH=/scratch/spf248/scratch_manu
  HOME_PATH=/home/spf248
elif [ ${SAM} -eq 0 ]
then
  SCRATCH_PATH=/scratch/mt4493
  HOME_PATH=/home/mt4493
fi

DATA_PATH=${SCRATCH_PATH}/twitter_labor/twitter-labor-data/data/${DATA_FOLDER}/${COUNTRY_CODE}

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
  elif [ ${MODEL_TYPE} = "roberta-base" ] || [ ${MODEL_TYPE} = "mrm8488/RuPERTa-base" ] || [ ${MODEL_TYPE} = "pysentimiento/robertuito-base-cased" ] || [ ${MODEL_TYPE} = "pysentimiento/robertuito-base-uncased" ]; then
    MODEL_NAME="roberta"
  elif [ ${MODEL_TYPE} = "xlm-roberta-base" ]; then
    MODEL_NAME="xlmroberta"
  fi
  echo ${MODEL_NAME}
  }


MODEL_NAME_1=$(select_model_name "${MODEL_TYPE_1}")

CODE_FOLDER=${SCRATCH_PATH}/twitter_labor/code/twitter/code/2-twitter_labor/2-model_training/bert_models

#echo "Launch series of training with different seeds";
#for i in {1..5}; do
#  for j in {3..6}; do
#    sbatch ${CODE_FOLDER}/train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_1} ${MODEL_TYPE_1} ${i} True ${j}
#  done
#done

echo "Launch series of training with different seeds";
for i in {1..15}; do
#  for j in {3..6}; do
  j=5
  sbatch ${CODE_FOLDER}/train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_1} ${MODEL_TYPE_1} ${i} True ${j}
done


#sbatch /scratch/mt4493/twitter_labor