import requests
import zipfile
import json
import io, os
import sys
import re
import socket
import pandas as pd
import reverse_geocoder as rg
import numpy as np
from glob import glob
import argparse
import boto3
import shutil


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--surveyId", type=str)
    parser.add_argument("--reject_bots", type=int, default=0)
    parser.add_argument("--HITId", type=str, default=None)
    parser.add_argument("--sam_API", type=int)
    parser.add_argument("--discard_x", type=int, default=3)
    parser.add_argument("--iteration_number", type=str)

    args = parser.parse_args()
    return args


def exportSurvey(apiToken, surveyId, dataCenter, fileFormat, path_to_data):
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
        os.path.join(path_to_data, surveyId))
    print('Complete')


def fill_assignment_worker_ids_dict(assignments_dict, assignment_worker_ids_dict):
    for assignment_nb in range(len(assignments_dict['Assignments'])):
        assignment_info_dict = assignments_dict['Assignments'][assignment_nb]
        assignment_worker_ids_dict[assignment_info_dict['WorkerId']] = assignment_info_dict['AssignmentId']
    return assignment_worker_ids_dict


if __name__ == "__main__":
    args = get_args_from_command_line()
    path_to_data = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics/{args.country_code}/iter{args.iteration_number}/labeling'
    if args.sam_API == 1:
        with open('/scratch/spf248/twitter/data/keys/qualtrics/apiToken', 'r') as f:
            apiToken = eval(f.readline())
    else:
        with open('/scratch/mt4493/twitter_labor/twitter-labor-data/data/keys/qualtrics/apiToken.txt', 'r') as f:
            apiToken = f.readline()
    print(apiToken)
    # Export Survey
    if os.path.exists(
            os.path.join(path_to_data, args.surveyId)):
        print("Overwriting existing folder")
        shutil.rmtree(os.path.join(path_to_data, args.surveyId), ignore_errors=True)

    if not re.compile('^SV_.*').match(args.surveyId):
        print("survey Id must match ^SV_.*")
    else:
        exportSurvey(apiToken=apiToken, surveyId=args.surveyId, dataCenter='nyu.ca1', fileFormat='csv',
                     path_to_data=path_to_data)
    file_path = \
        [file for file in glob(os.path.join(path_to_data, args.surveyId, '*.csv')) if 'labor-market-tweets' in file][0]
    # Analyse Results
    df = pd.read_csv(file_path, low_memory=False)
    # First two rows contain metadata
    df.drop([0, 1], inplace=True)

    df = df.loc[(df['QIDWorker'].dropna().drop_duplicates().index)].set_index('QIDWorker').copy()

    # places = rg.search(
    #    [tuple(x) for x in df[['LocationLatitude', 'LocationLongitude']].astype(float).dropna().values.tolist()])

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
    print('# Workers:', df.shape[0])

    # Checks
    checks = df[[col for col in df.columns if 'check' in col]].copy()
    checks.columns.name = 'QID'

    # Rearrange Results
    checks = checks.stack().rename('score').to_frame()

    # Extract Check ID
    checks['check_id'] = checks.index.get_level_values('QID').map(
        lambda x: re.findall('check-(\d)', x)[0])

    # Extract Class ID
    checks['class_id'] = checks.index.get_level_values('QID').map(
        lambda x: re.findall('_(\d)', x)[0])

    # Sort Values
    checks = checks.reset_index(level='QIDWorker').sort_values(
        by=['QIDWorker', 'check_id', 'class_id']).set_index(
        ['QIDWorker', 'check_id', 'class_id'])


    # Bot=Fail to give a Yes to the 2 check questions
    def is_bot(x):
        l = x.split('_')
        if len(l) == 10:
            if l[1] == '1' and l[4] == '2' and l[8] == '1' and l[9] == '2':
                if l[0] == '1' and l[2] == '2' and l[3] == '2' and l[5] == '2' and l[
                    6] == '2' and l[7] == '2':
                    return 3
                else:
                    return 2
            elif (l[1] == '1' and l[4] == '2') is not (l[8] == '1' and l[9] == '2'):
                return 1
        return 0


    bots = checks.unstack(
        level='check_id').unstack(
        level='class_id').fillna('').apply(
        lambda x: '_'.join(x), 1).apply(is_bot).where(
        lambda x: x == 0).dropna().index

    print('# Workers who failed both check questions (= bots?):', bots.shape[0])
    print('# Worker ID of workers who failed both check questions (= bots?):', bots)

    workers_1_question_right = checks.unstack(
        level='check_id').unstack(
        level='class_id').fillna('').apply(
        lambda x: '_'.join(x), 1).apply(is_bot).where(
        lambda x: x == 1).dropna().index

    print('# Workers who just passed one check:', workers_1_question_right.shape[0])
    
    workers_2_question_right = checks.unstack(
        level='check_id').unstack(
        level='class_id').fillna('').apply(
        lambda x: '_'.join(x), 1).apply(is_bot).where(
        lambda x: x == 2 ).dropna().index
    print('# Workers who passed the two check questions:', workers_2_question_right.shape[0])
    
    good_turkers = checks.unstack(
        level='check_id').unstack(
        level='class_id').fillna('').apply(
        lambda x: '_'.join(x), 1).apply(is_bot).where(
        lambda x: x == 3).dropna().index
    print('# Workers who answered all questions right for the two check blocks:', good_turkers.shape[0])

    bots_to_be_discarded = checks.unstack(
        level='check_id').unstack(
        level='class_id').fillna('').apply(
        lambda x: '_'.join(x), 1).apply(is_bot).where(
        lambda x: x < args.discard_x).dropna().index

    if args.reject_bots == 1:
        keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/keys/mturk'
        with open(os.path.join(keys_path, 'access_key_id.txt'), 'r') as f:
            access_key_id = f.readline().strip()

        with open(os.path.join(keys_path, 'secret_access_key.txt'), 'r') as f:
            secret_access_key = f.readline().strip()

        requester_feedback_dict = {
            'US': f'We are sorry to tell you that you have not passed the quality checks for HIT {args.HITId} (questions on English tweets). Therefore, we must reject your assignment. Thank you for your understanding',
            'MX': f'Lamentamos comunicarle que no ha superado los controles de calidad de HIT {args.HITId} (preguntas sobre los tweets en español). Por lo tanto, debemos rechazar su asignación. Gracias por su comprensión.',
            'BR': f'Lamentamos dizer que você não passou nos controles de qualidade do HIT {args.HITId} (perguntas sobre tweets portugueses). Portanto, devemos rejeitar sua tarefa. Obrigado por sua compreensão.'
        }

        mturk = boto3.client('mturk',
                             aws_access_key_id=access_key_id,
                             aws_secret_access_key=secret_access_key,
                             region_name='us-east-1',
                             endpoint_url='https://mturk-requester.us-east-1.amazonaws.com'
                             )

        # terminate HIT
        mturk.update_expiration_for_hit(
            HITId=args.HITId,
            ExpireAt=0
        )
        print('HIT was terminated')
        assignments_dict = mturk.list_assignments_for_hit(
            HITId=args.HITId,
        )
        assignment_worker_ids_dict = dict()
        while 'NextToken' in assignments_dict.keys():
            assignment_worker_ids_dict = fill_assignment_worker_ids_dict(assignments_dict=assignments_dict,
                                                                         assignment_worker_ids_dict=assignment_worker_ids_dict)
            assignments_dict = mturk.list_assignments_for_hit(
                HITId=args.HITId,
                NextToken=assignments_dict['NextToken']
            )

            if 'NextToken' not in assignments_dict.keys() and assignments_dict['NumResults'] > 0:
                assignment_worker_ids_dict = fill_assignment_worker_ids_dict(assignments_dict=assignments_dict,
                                                                             assignment_worker_ids_dict=assignment_worker_ids_dict)

        for bot_id in bots:
            try:
                mturk.associate_qualification_with_worker(
                    QualificationTypeId='3RDXJZR9A1H33MQ79TZZWYBXX8WCYD',
                    WorkerId=bot_id,
                    IntegerValue=1,
                    SendNotification=False)
                print(f'Assigned bot qualification to bot {bot_id}')
            except:
                print(f'Failed to assign bot qualification to bot {bot_id}')
            if bot_id in assignment_worker_ids_dict.keys():
                assignment_id = assignment_worker_ids_dict[bot_id]
                try:
                    mturk.reject_assignment(
                        AssignmentId=assignment_id,
                        RequesterFeedback=requester_feedback_dict[args.country_code]
                    )
                except:
                    print(f'Not able to reject assignment for bot {bot_id} ')
        print('Reject assignments for detected bots')

    # Remove checks
    df.drop([col for col in df.columns if 'check' in col], 1, inplace=True)
    df.columns.name = 'QID'

    # Rearrange Results
    df = df.stack().rename('score').to_frame()

    # Extract Tweets ID (Removing Extra Indexing)
    df['tweet_id'] = df.index.get_level_values('QID').map(
        lambda x: re.sub('-v\d', '', x.replace('ID_', '').replace('.1', '')).split('_')[0])

    # Extract Classes (Removing Extra Indexing)
    df['class_id'] = df.index.get_level_values('QID').map(
        lambda x: re.sub('-v\d', '', x.replace('ID_', '').replace('.1', '')).split('_')[1])

    # Sort Values
    df = df.reset_index(level='QIDWorker').sort_values(
        by=['tweet_id', 'class_id', 'QIDWorker']).set_index(
        ['tweet_id', 'class_id', 'QIDWorker'])

    # Drop users who have failed at least one check
    df.drop(bots_to_be_discarded, level='QIDWorker', inplace=True, errors='ignore')

    # Convert Scores
    df.score = df.score.apply(lambda x: {
        '1': 'yes',
        '2': 'no',
        '3': 'unsure'}[x])

    df.to_csv(
        os.path.join(path_to_data, args.surveyId, 'labels.csv'))
