import numpy as np
from glob import glob
import argparse
import os
import requests
import zipfile
import json
import io
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import reverse_geocoder as rg
import boto3


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="US")
    args = parser.parse_args()
    return args


def exportSurvey(apiToken, surveyId, dataCenter, fileFormat):
    surveyId = surveyId
    fileFormat = fileFormat
    dataCenter = dataCenter

    # Setting static parameters
    requestCheckProgress = 0.0
    progressStatus = "inProgress"
    baseUrl = "https://{0}.qualtrics.com/API/v3/responseexports/".format(dataCenter)
    headers = {
        "content-type": "application/json",
        "x-api-token": apiToken,
    }

    # Step 1: Creating Data Export
    downloadRequestUrl = baseUrl
    downloadRequestPayload = '{"format":"' + fileFormat + '","surveyId":"' + surveyId + '"}'
    downloadRequestResponse = requests.request("POST", downloadRequestUrl, data=downloadRequestPayload, headers=headers)
    progressId = downloadRequestResponse.json()["result"]['id']
    print(downloadRequestResponse.text)

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while progressStatus != "complete" and progressStatus != "failed":
        print("progressStatus=", progressStatus)
        requestCheckUrl = baseUrl + progressId
        requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
        requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
        print("Download is " + str(requestCheckProgress) + " complete")
        progressStatus = requestCheckResponse.json()["result"]["status"]

    # step 2.1: Check for error
    if progressStatus is "failed":
        raise Exception("export failed")

    # # Step 3: Downloading file
    requestDownloadUrl = baseUrl + progressId + '/file'
    requestDownload = requests.request("GET", requestDownloadUrl, headers=headers, stream=True)

    # Step 4: Unzipping the file
    zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall(
        os.path.join(path_to_data, country_code, "labeling", surveyId))
    print('Complete')


def select_paths(x):
    if x == 'labor-market-tweets.csv':
        return x
    elif any(xs in x for xs in ['US', 'BR', 'MX']):
        return x
    else:
        return None


if __name__ == '__main__':
    args = get_args_from_command_line()
    country_code = args.country_code
    path_to_data = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics'

    with open(os.path.join(path_to_data, 'keys/apiToken.txt'), 'r') as f:
        apiToken = f.readline()

    dataCenter = "nyu.ca1"
    fileFormat = "csv"

    survey_ids_list = glob(os.path.join(path_to_data, country_code, 'labeling', '*'))
    survey_ids_list = [survey_id for survey_id in survey_ids_list if survey_id != 'labels.pkl']
    print(survey_ids_list)

    worker_id_list = list()
    for surveyId in survey_ids_list:
        if not os.path.exists(os.path.join(path_to_data, country_code, 'labeling', surveyId)):
            if not re.compile('^SV_.*').match(surveyId):
                print("survey Id must match ^SV_.*")
            else:
                exportSurvey(apiToken, surveyId, dataCenter, fileFormat)
        folder_path = os.path.join(path_to_data, country_code, 'labeling', surveyId)
        file_path_list = os.listdir(folder_path)
        file_path_list = [select_paths(path) for path in file_path_list if select_paths(path) is not None]
        if len(file_path_list) > 0:
            df = pd.read_csv(os.path.join(folder_path, file_path_list[0]), low_memory=False)

            # First two rows contain metadata
            df.drop([0, 1], inplace=True)

            df = df.loc[(df['QIDWorker'].dropna().drop_duplicates().index)].set_index('QIDWorker').copy()

            # places=rg.search([tuple(x) for x in df[['LocationLatitude','LocationLongitude']].astype(float).dropna().values.tolist()])

            print('# of workers who refused the consent form:', (df.QIDConsent.astype(int) == 0).sum())
            print('# of workers who did not complete the survey:', (df.Finished.astype(int) == 0).sum())

            to_drop = [
                'ResponseID',
                'ResponseSet',
                'IPAddress',
                'StartDate',
                'EndDate',
                'RecipientLastName',
                'RecipientFirstName',
                'RecipientEmail',
                'ExternalDataReference',
                'Finished',
                'Status',
                'Random ID',
                'QIDConsent',
                'QIDDescription',
                'QIDCompletion',
                'LocationLatitude',
                'LocationLongitude',
                'LocationAccuracy']

            df.drop(to_drop, 1, inplace=True, errors='ignore')
            df.drop([x for x in df.columns if 'BR-FL_' in x], 1, inplace=True, errors='ignore')
            df = df.reset_index()
            df = df[df['QIDWorker'].notna()].reset_index(drop=True)
            worker_id_list = worker_id_list + df['QIDWorker'].tolist()

    print(len(worker_id_list))

    # Remove duplicates
    worker_id_list = list(dict.fromkeys(worker_id_list))
    # Assign qualification

    keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/mturk/keys'
    with open(os.path.join(keys_path, 'access_key_id.txt'), 'r') as f:
        access_key_id = f.readline().strip()

    with open(os.path.join(keys_path, 'secret_access_key.txt'), 'r') as f:
        secret_access_key = f.readline().strip()

    mturk = boto3.client('mturk',
                         aws_access_key_id=access_key_id,
                         aws_secret_access_key=secret_access_key,
                         region_name='us-east-1',
                         endpoint_url='https://mturk-requester.us-east-1.amazonaws.com'
                         )

    for worker_id in worker_id_list:
        try:
            mturk.associate_qualification_with_worker(
                QualificationTypeId='3YLTB9JB8TED72KIAHT6K4NASKY63F',
                WorkerId=worker_id,
                IntegerValue=1,
                SendNotification=False)
        except:
            print(worker_id)

    print("The Qualification was assigned to all workers who already completed the US survey (including bots). ")
