#!/bin/bash
#SBATCH --job-name=training_glove_cnn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_training_glove_cnn_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

DATA_FOLDER=$1
TIMESTAMP=$(date +%s)
JOB_ID=${SLURM_JOB_ID}

SLASH=/

if [[ ${DATA_FOLDER} == *"/"* ]]
then
    DATA_FOLDER_WITHOUT_SLASH=${DATA_FOLDER//[${SLASH}]/_}
else
    DATA_FOLDER_WITHOUT_SLASH=${DATA_FOLDER}
fi

HOLDOUT_DATA_FOLDER=/scratch/mt4493/twitter_labor/twitter-labor-data/data/jul23_iter0/preprocessed
GLOVE_FILES_PATH=/scratch/mt4493/twitter_labor/code/twitter/code/twitter_labor/2-model_training/glove_cnn_models/glove_files
# Load packages and activate virtual environment
module purge
module load anaconda3/2019.10
source /scratch/mt4493/twitter_labor/code/envs/glove_cnn_training_env/bin/activate
echo "virtualenv activated"

#create output folder
mkdir /scratch/mt4493/twitter_labor/trained_models/"GloVe_CNN"_"${DATA_FOLDER_WITHOUT_SLASH}"_"${JOB_ID}"
OUTPUT_DIR=/scratch/mt4493/twitter_labor/trained_models/"GloVe_CNN"_"${DATA_FOLDER_WITHOUT_SLASH}"_"${JOB_ID}"
echo "Output folder created: ${OUTPUT_DIR}"

cd /scratch/mt4493/twitter_labor/code/twitter/code/twitter_labor/2-model_training/glove_cnn_models

echo '***********************STARTING TRAINING ON LABEL lost_job_1mo***************************************************'
python3 train_glove_cnn.py \
 --num_epochs 100 \
 --evaluate_every 10 \
 --training_data_path "${DATA_FOLDER}" \
 --holdout_data_path ${HOLDOUT_DATA_FOLDER} \
 --embeddings_path ${GLOVE_FILES_PATH}/embeddings.npy \
 --vocab_path ${GLOVE_FILES_PATH}/vocab.pckl \
 --label lost_job_1mo \
 --output_dir ${OUTPUT_DIR}/lost_job_1mo \
 --slurm_job_timestamp "${TIMESTAMP}" \
 --slurm_job_id "${JOB_ID}"

echo '***********************DONE TRAINING ON LABEL lost_job_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_unemployed***************************************************'
python3 train_glove_cnn.py \
 --num_epochs 100 \
 --evaluate_every 10 \
 --training_data_path "${DATA_FOLDER}" \
 --holdout_data_path ${HOLDOUT_DATA_FOLDER} \
 --embeddings_path ${GLOVE_FILES_PATH}/embeddings.npy \
 --vocab_path ${GLOVE_FILES_PATH}/vocab.pckl \
 --label is_unemployed \
 --output_dir ${OUTPUT_DIR}/is_unemployed \
 --slurm_job_timestamp "${TIMESTAMP}" \
 --slurm_job_id "${JOB_ID}"
echo '***********************DONE TRAINING ON LABEL is_unemployed*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_search***************************************************'
python3 train_glove_cnn.py \
 --num_epochs 100 \
 --evaluate_every 10 \
 --training_data_path "${DATA_FOLDER}" \
 --holdout_data_path ${HOLDOUT_DATA_FOLDER} \
 --embeddings_path ${GLOVE_FILES_PATH}/embeddings.npy \
 --vocab_path ${GLOVE_FILES_PATH}/vocab.pckl \
 --label job_search \
 --output_dir ${OUTPUT_DIR}/job_search \
 --slurm_job_timestamp "${TIMESTAMP}" \
 --slurm_job_id "${JOB_ID}"
echo '***********************DONE TRAINING ON LABEL job_search*******************************************************'

echo '***********************STARTING TRAINING ON LABEL is_hired_1mo***************************************************'
python3 train_glove_cnn.py \
 --num_epochs 100 \
 --evaluate_every 10 \
 --training_data_path "${DATA_FOLDER}" \
 --holdout_data_path ${HOLDOUT_DATA_FOLDER} \
 --embeddings_path ${GLOVE_FILES_PATH}/embeddings.npy \
 --vocab_path ${GLOVE_FILES_PATH}/vocab.pckl \
 --label is_hired_1mo \
 --output_dir ${OUTPUT_DIR}/is_hired_1mo \
 --slurm_job_timestamp "${TIMESTAMP}" \
 --slurm_job_id "${JOB_ID}"
echo '***********************DONE TRAINING ON LABEL is_hired_1mo*******************************************************'

echo '***********************STARTING TRAINING ON LABEL job_offer***************************************************'
python3 train_glove_cnn.py \
 --num_epochs 100 \
 --evaluate_every 10 \
 --training_data_path "${DATA_FOLDER}" \
 --holdout_data_path ${HOLDOUT_DATA_FOLDER} \
 --embeddings_path ${GLOVE_FILES_PATH}/embeddings.npy \
 --vocab_path ${GLOVE_FILES_PATH}/vocab.pckl \
 --label job_offer \
 --output_dir ${OUTPUT_DIR}/job_offer \
 --slurm_job_timestamp "${TIMESTAMP}" \
 --slurm_job_id "${JOB_ID}"
echo '***********************DONE TRAINING ON LABEL job_offer*******************************************************'

#srun time python -u 8.2-random-samples-UNDERsampled-separate-labels.py job_search > /scratch/da2734/twitter/running_on_200Msamples/array_logs/job_search/8.2-random-samples-UNDERsampled-separate-labels_${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
exit
