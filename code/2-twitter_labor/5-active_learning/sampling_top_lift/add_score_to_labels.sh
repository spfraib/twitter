#!/bin/bash

module load python/gnu/3.6.5
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8

CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/sampling_top_lift
TIMESTAMP=$(date +%s)
JOB_NAME=add_score_to_labels_${TIMESTAMP}
spark-submit --master yarn --deploy-mode cluster --name ${JOB_NAME} \
  --conf spark.yarn.submit.waitAppCompletion=false --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.speculation=false --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
  --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores 10 \
  --driver-memory 10G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
  --executor-cores 10 --executor-memory 10G \
  ${CODE_FOLDER}/add_score_to_labels.py \


echo "Submitted Spark job"