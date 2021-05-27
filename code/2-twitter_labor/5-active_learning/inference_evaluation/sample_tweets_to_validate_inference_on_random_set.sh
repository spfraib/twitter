#!/bin/bash

COUNTRY_CODE=$1
MODEL_FOLDER=$2

NB_CORES=20

module load python/gnu/3.6.5
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/inference_evaluation
TIMESTAMP=$(date +%s)
JOB_NAME=sample_tweets_to_validate_inference_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores ${NB_CORES} \
  --driver-memory ${NB_CORES}G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores ${NB_CORES} --executor-memory ${NB_CORES}G \
  ${CODE_FOLDER}/sample_tweets_to_validate_inference_on_random_set.py \
  --country_code ${COUNTRY_CODE} \
  --model_folder ${MODEL_FOLDER}
