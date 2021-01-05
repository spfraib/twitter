#!/bin/bash

COUNTRY_CODE=US
#SBATCH --job-name=operations
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

module purge
module load anaconda3/2020.02

source /scratch/mt4493/twitter_labor/code/envs/inference_env/bin/activate

cd /scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation


SURVEY_ID_LIST="SV_7aeWaaTnG7NiYOV,SV_8uMuwiJVgsGDPjn,SV_81Z6plk4o7m4k0R,SV_6mSgd3aLStLTkcl,SV_d6Z7MnJPXSPYnMp,SV_dhDa4Jrlt5rEA7j,SV_b7rYWRVD9CE04dL,SV_9GGdQAjHupi5OKx,SV_eQdmwpIo95tdItD,SV_6JZyroFZpUizsjj,SV_54hSt6qDYsbKAxn,SV_bkhY4hq3qfNnWap,SV_0ClQjqjqerIqZNj,SV_8Dk9huGrMrtFesZ,SV_9TzzayDad3RgIXX,SV_3CrEKpGhqlrO8pT,SV_1FEeJM9n3Pi8Azr,SV_agTJ6PT6XgKqRCd,SV_0dB80s8q5OhAV8x,SV_1X0kskPK25dnt6l,SV_7V75YTlrECFweKV,SV_ctODgZS3rLY5rz7,SV_9FWQ2zw1kp5gXIx,SV_8iXk8NykLgCQEIt,SV_cvxRxMs5UNulogd,SV_er3tETgYDXv1G85,SV_6ydfWA2LrVqOCBD,SV_1MRHgIP6EZSuXWt,SV_9FQXSDKa50C8iKp,SV_3fsqdhfXVkxsaSV,SV_5mv8DI1N0sXTZgp,SV_24RoQ3TAAnEpaN7,SV_7aFs0rDpHMfX4ah,SV_4YjcoEjVDDreyrP,SV_0Ilb9QkeyHziljT"

IFS=,

for SURVEY_ID in ${SURVEY_ID_LIST};
do
echo "******** Getting results for ${SURVEY_ID} ********"
python3 qualtrics/get_results_from_qualtrics_sam.py \
--country_code ${COUNTRY_CODE} \
--surveyId ${SURVEY_ID} \

done

echo '******** Merging results from Qualtrics ********'
python3 qualtrics/merge-all-qualtrics-results.py \
--country_code ${COUNTRY_CODE}
