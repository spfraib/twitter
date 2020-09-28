#!/bin/bash

INFERENCE_FOLDER=$1

module load python/gnu/3.6.5
module load spark/2.4.0
export PYSPARK_PYTHON=/share/apps/python/3.6.5/bin/python
export PYSPARK_DRIVER_PYTHON=/share/apps/python/3.6.5/bin/python
export PYTHONIOENCODING=utf8

hdfs dfs -put /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output /user/mt4493/twitter/inference/${INFERENCE_FOLDER}
echo "Loaded inference data on Hadoop. Launching the PySpark script"
CODE_FOLDER=/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/5-active_learning/preliminary

spark-submit --master yarn --deploy-mode cluster --conf spark.yarn.submit.waitAppCompletion=false \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer --conf spark.speculation=false \
--conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 --conf spark.executorEnv.LANG=en_US.UTF-8 --driver-cores 10 \
--driver-memory 10G --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=50 \
--executor-cores 10 --executor-memory 10G \
${CODE_FOLDER}/process_inference_output_spark.py \
--inference_output_folder /user/mt4493/twitter/inference/${INFERENCE_FOLDER} \
--random_chunks_with_operations_folder /user/mt4493/twitter/random_chunks_with_operations

echo "PySpark script done. Copying output data to scratch."
hdfs dfs -get /user/mt4493/twitter/inference/${INFERENCE_FOLDER}/joined /scratch/mt4493/twitter_labor/twitter-labor-data/data/inference/${INFERENCE_FOLDER}/output/joined
echo "Copying data finished."