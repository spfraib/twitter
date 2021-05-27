#!/bin/bash

INFERENCE_FOLDER=$1

NB_CORES=20

module load python/gcc/3.7.9
module load spark/3.0.1
#export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
#export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
#export PYTHONIOENCODING=utf8

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/adaptive_retrieval
TIMESTAMP=$(date +%s)
JOB_NAME=retrieve_top_tweets_peel_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores ${NB_CORES} \
  --driver-memory ${NB_CORES}G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores ${NB_CORES} --executor-memory ${NB_CORES}G \
  ${CODE_FOLDER}/retrieve_top_tweets_peel.py \
  --inference_folder ${INFERENCE_FOLDER}
