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
        os.path.join(path_to_data, "classification", country_code, "labeling", 'qualtrics', surveyId))
    print('Complete')

def select_paths(x):
    if x == 'labor-market-tweets.csv':
        return x
    elif 'US' in x:
        return x
    else:
        return None

country_code = 'US'
path_to_data = '/scratch/spf248/twitter/data'

with open(os.path.join(path_to_data,'keys/qualtrics/apiToken'),'r') as f:
    apiToken = eval(f.readline())

dataCenter = "nyu.ca1"
fileFormat = "csv"

survey_ids_list = glob(os.path.join(path_to_data,'classification',country_code,'labeling','qualtrics','*'))

print(survey_ids_list)

worker_id_list = list()
for surveyId in survey_ids_list:
    if not os.path.exists(os.path.join(path_to_data,"classification",country_code,"labeling",'qualtrics',surveyId)):
        if not re.compile('^SV_.*').match(surveyId):
            print("survey Id must match ^SV_.*")
        else:
            exportSurvey(apiToken, surveyId, dataCenter, fileFormat)
    folder_path = os.path.join(path_to_data,"classification",country_code,"labeling",'qualtrics',surveyId)
    file_path_list = os.listdir(folder_path)
    file_path_list = [select_paths(path) for path in file_path_list if select_paths(path) is not None]
    if len(file_path_list) > 0:
        df=pd.read_csv(os.path.join(folder_path, file_path_list[0]),low_memory=False)
    
        # First two rows contain metadata
        df.drop([0,1],inplace=True)

        df=df.loc[(df['QIDWorker'].dropna().drop_duplicates().index)].set_index('QIDWorker').copy()

        #places=rg.search([tuple(x) for x in df[['LocationLatitude','LocationLongitude']].astype(float).dropna().values.tolist()])

        print('# of workers who refused the consent form:', (df.QIDConsent.astype(int)==0).sum())
        print('# of workers who did not complete the survey:', (df.Finished.astype(int)==0).sum())

        to_drop=[
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

        df.drop(to_drop,1,inplace=True,errors='ignore')
        df.drop([x for x in df.columns if 'BR-FL_' in x],1,inplace=True,errors='ignore')
        df = df.reset_index()
        worker_id_list = worker_id_list +  df['QIDWorker'].tolist()

print(len(worker_id_list))
