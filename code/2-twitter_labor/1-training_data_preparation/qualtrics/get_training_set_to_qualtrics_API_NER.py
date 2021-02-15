import socket
import pandas as pd
from timeit import default_timer as timer
import os
import requests
import json
import numpy as np
import pyarrow.parquet as pq
from glob import glob
from datetime import datetime
import argparse
from ekphrasis.classes.preprocessor import TextPreProcessor
import re


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--n_workers", type=int, help="number of workers",
                        default=20)
    parser.add_argument("--block_size", type=int, help="number of tweets per worker",
                        default=50)
    parser.add_argument("--version_number", type=str)
    args = parser.parse_args()
    return args


def create_survey(SurveyName, apiToken, dataCenter, language):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions".format(
        dataCenter)

    headers = {
        "x-api-token": apiToken,
        "content-type": "application/json",
        "Accept": "application/json"
    }

    data = {
        "SurveyName": SurveyName,
        "Language": language,
        "ProjectCategory": "CORE"
    }

    response = requests.post(baseUrl, json=data, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    SurveyID = json.loads(response.text)['result']['SurveyID']
    DefaultBlockID = json.loads(response.text)['result']['DefaultBlockID']

    return SurveyID, DefaultBlockID


def get_options(SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/options".format(
        dataCenter, SurveyID)

    headers = {
        "x-api-token": apiToken,
    }

    response = requests.get(baseUrl, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    return json.loads(response.text)["result"]


def update_options(SurveyOptions, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/options".format(
        dataCenter, SurveyID)

    headers = {
        'accept': "application/json",
        "content-type": "application/json",
        "x-api-token": apiToken,
    }

    response = requests.put(baseUrl, json=SurveyOptions, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])


def get_flow(SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/flow".format(
        dataCenter, SurveyID)

    headers = {
        "x-api-token": apiToken,
    }

    response = requests.get(baseUrl, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    return json.loads(response.text)["result"]


def update_flow(SurveyFlow, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/flow".format(
        dataCenter, SurveyID)

    headers = {
        'accept': "application/json",
        "content-type": "application/json",
        "x-api-token": apiToken,
    }

    response = requests.put(baseUrl, json=SurveyFlow, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])


def create_block(BlockName, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/blocks".format(
        dataCenter, SurveyID)

    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        "x-api-token": apiToken,
    }

    BlockTemplate = {
        "Type": "Standard",
        "Description": BlockName,
    }

    response = requests.post(baseUrl, json=BlockTemplate, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    BlockID = json.loads(response.text)['result']['BlockID']
    FlowID = json.loads(response.text)['result']['FlowID']

    return BlockID, FlowID


def get_block(BlockID, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/blocks/{2}".format(
        dataCenter, SurveyID, BlockID)

    headers = {
        "x-api-token": apiToken,
    }

    response = requests.get(baseUrl, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    return json.loads(response.text)["result"]


def update_block(BlockData, BlockID, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/blocks/{2}".format(
        dataCenter, SurveyID, BlockID)

    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        "x-api-token": apiToken,
    }

    response = requests.put(baseUrl, json=BlockData, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])


def create_question(QuestionData, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/questions".format(
        dataCenter, SurveyID)

    headers = {
        # 'accept': "application/json",
        # 'content-type': "application/json",
        "x-api-token": apiToken,
    }

    response = requests.post(baseUrl, json=QuestionData, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    return json.loads(response.text)['result']['QuestionID']


def get_question(QuestionID, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/questions/{2}".format(
        dataCenter, SurveyID, QuestionID)

    headers = {
        "x-api-token": apiToken,
    }

    response = requests.get(baseUrl, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])

    return json.loads(response.text)["result"]


def update_question(QuestionData, QuestionID, SurveyID, apiToken, dataCenter):
    baseUrl = "https://{0}.qualtrics.com/API/v3/survey-definitions/{1}/questions/{2}".format(
        dataCenter, SurveyID, QuestionID)

    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        "x-api-token": apiToken,
    }

    response = requests.put(baseUrl, json=QuestionData, headers=headers)

    if json.loads(response.text)["meta"]["httpStatus"] != '200 - OK':
        print(json.loads(response.text)["meta"]["httpStatus"])


def discard_already_labelled_tweets(path_to_labelled, to_label_df):
    labels_path = os.path.join(path_to_labelled, 'labels.pkl')
    if os.path.isfile(labels_path):
        df = pd.read_pickle(labels_path)
        df = df[['tweet_id']]
        df = df.drop_duplicates().reset_index(drop=True)
        list_labelled_tweet_ids = df['tweet_id'].tolist()
        to_label_df = to_label_df[~to_label_df['tweet_id'].isin(list_labelled_tweet_ids)].reset_index(drop=True)
        return to_label_df
    else:
        return to_label_df

    # df_list = list()
    # for folder in os.listdir(path_to_labelled):
    #     for file in glob(os.path.join(path_to_labelled, folder, '*.csv')):
    #         df = pd.read_csv(file)
    #         if 'tweet_id' in df.columns:
    #             df = df[['tweet_id']]
    #             df = df.set_index(['tweet_id'])
    #             df_list.append(df)
    # if len(df_list) > 0:
    #     final_df = pd.concat(df_list)
    #     final_df = final_df.reset_index()
    #     final_df = final_df.drop_duplicates().reset_index(drop=True)
    #     list_labelled_tweet_ids = final_df['tweet_id'].tolist()
    #     to_label_df = to_label_df[~to_label_df['tweet_id'].isin(list_labelled_tweet_ids)].reset_index(drop=True)
    #     return to_label_df
    # else:
    #     return to_label_df


def tweet_preprocessing(tweet):
    tweet = tweet.replace('<hashtag>', '#')
    tweet = tweet.replace('</hashtag>', '')
    tweet = tweet.replace('  ', ' ')
    tweet = tweet.replace('-', ' - ')
    tweet = re.sub('http\S+', 'HTTPURL', tweet)
    tweet = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '@USER', tweet)
    tweet = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", tweet)
    return tweet


def make_choices_dict_from_tweet(tweet, max_past_index):
    word_list = tweet.split(' ')
    index = max_past_index + 1
    position_token = 1
    choices_dict = dict()
    hashtag_indices_dict = dict()
    for word in word_list:
        choices_dict[str(index)] = {
            'WordLength': len(word),
            'Word': word,
            'Display': f'{str(position_token)}: {word}'
        }
        if index == max_past_index + 1:
            choices_dict[str(index)]['WordIndex'] = 0
        else:
            choices_dict[str(index)]['WordIndex'] = 1 + choices_dict[str(index - 1)]['WordIndex'] + \
                                                    choices_dict[str(index - 1)]['WordLength']
        if word == "#":
            hashtag_indices_dict[str(position_token - 1)] = True
        index = index + 1
        position_token = position_token + 1
    return choices_dict, hashtag_indices_dict


if __name__ == "__main__":
    args = get_args_from_command_line()
    if 'manuto' in socket.gethostname().lower():
        path_to_data = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/job_offer_NER/test'
    else:
        path_to_data = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/job_offer_NER/{args.country_code}/test'
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    # Setting user Parameters
    # with open('/scratch/mt4493/twitter_labor/twitter-labor-data/data/keys/qualtrics/apiToken.txt', 'r') as f:
    #     apiToken = f.readline()
    apiToken = 'lHNO91YF14BUFvPrdosG5IxzzCUt6Y65SxZGvmH7'

    dataCenter = "nyu.ca1"
    SurveyName = f"ner-job-offer-tweets_{args.country_code}_it0_{args.n_workers}_workers_{args.block_size}_block_size_v{args.version_number}"
    SurveySourceID_dict = {
        'US': 'SV_1KMR8bkQiVFQ0zH',
    'MX': 'SV_1AYw7rmB7faPJR3',}
    # 'BR': 'SV_e9Xsw1ZtEvBX4jP'}
    SurveySourceID = SurveySourceID_dict[args.country_code]
    QuestionTemplateID = "QID1"
    QuestionConsentID = "QID2"
    QuestionWorkerID = "QID3"
    QuestionCompletionID = "QID4"
    QuestionDescriptionID = "QID5"
    QuestionFeedbackID = "QID7"
    country_language_dict = {
        'US': 'EN',
        'MX': 'ES',
        'BR': 'PT-BR'}
    survey_language = country_language_dict[args.country_code]
    country_code = args.country_code
    n_workers = args.n_workers
    block_size = args.block_size
    print(country_code)
    print('# n_workers:', n_workers)
    print('block_size:', block_size)
    checks_dict = {
        'US': ['Software Engineer job at Amazon in New York, NY',
               'Looking for a job? Check out this opportunity at Microsoft in Seattle, WA'],
        'MX': ['***********VACANTE DE EMPLEO*********** PARA: AQUADIVERSIONES CORRAL GRANDE SA DE CV; PUESTO: VENTAS DE MOSTRADOR; UBICACION: CDMX', '# EMPLEO Se busca Diseñador Gráfico Sr. #Monterrey +3 años como senior. Esencial que haya trabajado en agencias digitales y conozca de medios digitales. '],
        'BR': ['Perdi o meu emprego hoje.', 'Fui contratado hoje.']}
    checks_list = checks_dict[args.country_code]

    n_tweets = n_workers * (block_size - len(checks_list)) // 2
    print('# Tweets (2 workers per tweets + 2 attention checks):', n_tweets)

    # path to labelling as argument?
    # tweets = pq.ParquetDataset(
    #     glob(os.path.join(path_to_data, '*.parquet'))).read().to_pandas()
    tweets = pd.read_csv(os.path.join(path_to_data, 'test.csv'))

    # tweets = discard_already_labelled_tweets(
    #     path_to_labelled=f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/qualtrics/{args.country_code}/labeling',
    #     to_label_df=tweets)

    # preprocess tweets
    segmenter_path_dict = {'US': 'twitter', 'MX': '/scratch/mt4493/twitter_labor/code/repos_annexe/ekphrasis/ekphrasis/stats/twitter_MX',
                      'BR': '/scratch/mt4493/twitter_labor/code/repos_annexe/ekphrasis/ekphrasis/stats/twitter_BR'}
    text_processor = TextPreProcessor(annotate={"hashtag"}, segmenter=segmenter_path_dict[args.country_code], unpack_hashtags=True)


    def hashtag_segmentation(text):
        return "".join(text_processor.pre_process_doc(text))


    tweets['text'] = tweets['text'].apply(hashtag_segmentation)
    tweets['text'] = tweets['text'].apply(tweet_preprocessing)
    print(tweets['text'].head())
    # # TEMPORARY
    import decimal

    # create a new context for this task
    ctx = decimal.Context()

    # 20 digits should be enough for everyone :D
    ctx.prec = 20


    def float_to_str(f):
        """
        Convert the given float to a string,
        without resorting to scientific notation
        """
        d1 = ctx.create_decimal(repr(f))
        return format(d1, 'f')


    tweets['tweet_id'] = tweets['tweet_id'].apply(float_to_str)

    #tweets = tweets.sample(n=n_tweets, random_state=0)
    print('# Unique Tweets:', tweets.drop_duplicates('tweet_id').shape[0])

    tweets_0 = tweets.sample(frac=1, random_state=0).set_index('tweet_id')['text']
    tweets_0.index = tweets_0.index.map(lambda x: x + '-v0')
    tweets_1 = tweets.sample(frac=1, random_state=1).set_index('tweet_id')['text']
    tweets_1.index = tweets_1.index.map(lambda x: x + '-v1')

    # Split tweets into chunks with two labels per tweet
    chunks = np.array_split(pd.concat([tweets_0, tweets_1]), n_workers)

    # Add Attention Checks
    tweets_to_label = pd.concat([chunk.append(pd.Series({
        'check-0-worker-' + str(i): checks_list[0],
        'check-1-worker-' + str(i): checks_list[1]})).sample(frac=1, random_state=0)
                                 for i, chunk in enumerate(chunks)])

    print('Create New Survey')
    start = timer()

    SurveyID, BlockID = create_survey(SurveyName=SurveyName, apiToken=apiToken, dataCenter=dataCenter,
                                      language=survey_language)

    print("Done in", round(timer() - start), "sec")

    QuestionTemplateData = get_question(QuestionID=QuestionTemplateID, SurveyID=SurveySourceID, apiToken=apiToken,
                                        dataCenter=dataCenter)

    start = timer()
    print("Create Questions")

    for i, (tweet_id, tweet) in enumerate(tweets_to_label.iteritems()):

        if i % block_size == 0:
            BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
            BlockData['Type'] = 'Standard'
            update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken,
                         dataCenter=dataCenter)

            print('Block', i // block_size + 1)
            BlockID, FlowID = create_block("Worker " + str(i // block_size + 1), SurveyID=SurveyID, apiToken=apiToken,
                                           dataCenter=dataCenter)

            BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
            BlockData['Type'] = 'Default'
            update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken,
                         dataCenter=dataCenter)

        text_dict = {'US': 'Please label the following tweet:',
                     'MX': 'Por favor, etiquete el siguiente tweet:',
                     'BR': 'Favor rotular o seguinte tweet:'}
        text = text_dict[args.country_code]
        QuestionID = create_question(QuestionData=QuestionTemplateData, SurveyID=SurveyID, apiToken=apiToken,
                                     dataCenter=dataCenter)
        QuestionData = get_question(QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
        QuestionData['QuestionText'] = text
        QuestionData['QuestionDescription'] = text
        QuestionData['QuestionText_Unsafe'] = text
        QuestionData['DataExportTag'] = 'ID_' + tweet_id
        max_choice_order = max(QuestionData['ChoiceOrder'])
        choices_dict, hashtag_indices_dict = make_choices_dict_from_tweet(tweet=tweet, max_past_index=max_choice_order)
        QuestionData['Choices'] = choices_dict
        QuestionData['ChoiceOrder'] = [i for i in
                                       range(max_choice_order + 1, max_choice_order + 1 + len(QuestionData['Choices']))]
        QuestionData['WordChoiceIds'] = QuestionData['ChoiceOrder']
        QuestionData['HighlightText'] = f'{tweet}<br>'
        QuestionData['ExcludedWords'] = hashtag_indices_dict
        update_question(QuestionData=QuestionData, QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken,
                        dataCenter=dataCenter)

        if i % block_size == 0:
            BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
            BlockData['Options'] = {
                "BlockLocking": "false",
                "RandomizeQuestions": "false",
                "BlockVisibility": "Collapsed",
            }
            update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken,
                         dataCenter=dataCenter)

    print("Done in", round(timer() - start), "sec")

    BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    BlockData['Type'] = 'Standard'
    update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    print('Create Completion Block')
    BlockID, FlowID = create_block(BlockName="Completion", SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    BlockData['Type'] = 'Default'
    update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken,
                 dataCenter=dataCenter)

    print('Create Feedback Question')
    QuestionFeedbackData = get_question(QuestionID=QuestionFeedbackID, SurveyID=SurveySourceID, apiToken=apiToken,
                                          dataCenter=dataCenter)
    QuestionID = create_question(QuestionData=QuestionFeedbackData, SurveyID=SurveyID, apiToken=apiToken,
                                 dataCenter=dataCenter)
    QuestionData = get_question(QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    QuestionData['DataExportTag'] = 'QIDFeedback'
    update_question(QuestionData=QuestionData, QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken,
                    dataCenter=dataCenter)


    print('Create Completion Question')
    QuestionCompletionData = get_question(QuestionID=QuestionCompletionID, SurveyID=SurveySourceID, apiToken=apiToken,
                                          dataCenter=dataCenter)
    QuestionID = create_question(QuestionData=QuestionCompletionData, SurveyID=SurveyID, apiToken=apiToken,
                                 dataCenter=dataCenter)
    QuestionData = get_question(QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    QuestionData['DataExportTag'] = 'QIDCompletion'
    update_question(QuestionData=QuestionData, QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken,
                    dataCenter=dataCenter)

    print('Close Block')
    BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    BlockData['Options'] = {
        "BlockLocking": "false",
        "RandomizeQuestions": "false",
        "BlockVisibility": "Collapsed",
    }
    update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    SurveyFlow = get_flow(SurveyID, apiToken=apiToken,
                          dataCenter=dataCenter)

    print('Randomize Survey Flow')
    # Create a Randomizer Drawing One Block At Random Except Intro And Completion Block
    Randomizer = {
        'Type': 'BlockRandomizer',
        'FlowID': 'FL_' + str(max([int(el['FlowID'].split('_')[1]) for el in SurveyFlow['Flow']]) + 1),
        'SubSet': '1',
        'EvenPresentation': True,
        'Flow': SurveyFlow['Flow'][1:-1]}

    SurveyFlow['Flow'] = [
        SurveyFlow['Flow'][0],
        Randomizer,
        SurveyFlow['Flow'][-1],
    ]

    SurveyFlow['Properties']['Count'] += 1
    SurveyFlow['Properties'].update({
        'RemovedFieldsets': []})

    print('Embbeded Worker ID')
    EmbeddedData = {
        'Type': 'EmbeddedData',
        'FlowID': 'FL_' + str(max([int(el['FlowID'].split('_')[1]) for el in SurveyFlow['Flow']]) + 1),
        'EmbeddedData': [{
            'Description': 'Random ID',
            'Type': 'Custom',
            'Field': 'Random ID',
            'VariableType': 'String',
            'DataVisibility': [],
            'AnalyzeText': False,
            'Value': '${rand://int/1000000000:9999999999}'}]}

    SurveyFlow['Flow'] = [EmbeddedData] + SurveyFlow['Flow']
    SurveyFlow['Properties']['Count'] += 1

    update_flow(SurveyFlow=SurveyFlow, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    # Switch Default Block From Current ...
    BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    BlockData['Type'] = 'Standard'
    update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    # ... to Intro
    BlockID = SurveyFlow['Flow'][1]['ID']
    BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    BlockData['Type'] = 'Default'
    update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    print('Add Consent Question')
    QuestionConsentData = get_question(QuestionID=QuestionConsentID, SurveyID=SurveySourceID, apiToken=apiToken,
                                       dataCenter=dataCenter)
    QuestionID = create_question(QuestionData=QuestionConsentData, SurveyID=SurveyID, apiToken=apiToken,
                                 dataCenter=dataCenter)
    QuestionData = get_question(QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    QuestionData['DataExportTag'] = 'QIDConsent'
    update_question(QuestionData=QuestionData, QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken,
                    dataCenter=dataCenter)

    print('Add Worker ID Question')
    QuestionWorkerData = get_question(QuestionID=QuestionWorkerID, SurveyID=SurveySourceID, apiToken=apiToken,
                                      dataCenter=dataCenter)
    QuestionID = create_question(QuestionData=QuestionWorkerData, SurveyID=SurveyID, apiToken=apiToken,
                                 dataCenter=dataCenter)
    QuestionData = get_question(QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    QuestionData['DataExportTag'] = 'QIDWorker'
    update_question(QuestionData=QuestionData, QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken,
                    dataCenter=dataCenter)

    print('Add Description Question')
    QuestionDescriptionData = get_question(QuestionID=QuestionDescriptionID, SurveyID=SurveySourceID, apiToken=apiToken,
                                           dataCenter=dataCenter)
    QuestionID = create_question(QuestionData=QuestionDescriptionData, SurveyID=SurveyID, apiToken=apiToken,
                                 dataCenter=dataCenter)
    QuestionData = get_question(QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    QuestionData['DataExportTag'] = 'QIDDescription'
    update_question(QuestionData=QuestionData, QuestionID=QuestionID, SurveyID=SurveyID, apiToken=apiToken,
                    dataCenter=dataCenter)

    print('Close Intro Block')
    BlockData = get_block(BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)
    BlockData['Options'] = {
        "BlockLocking": "false",
        "RandomizeQuestions": "false",
        "BlockVisibility": "Collapsed",
    }
    BlockData['Description'] = 'Intro'
    update_block(BlockData=BlockData, BlockID=BlockID, SurveyID=SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    SurveyOptions = get_options(SurveyID, apiToken=apiToken, dataCenter=dataCenter)

    SurveyOptions.update({
        'BackButton': 'false',
        'SaveAndContinue': 'true',
        'SurveyProtection': 'PublicSurvey',
        'BallotBoxStuffingPrevention': 'true',
        'NoIndex': 'Yes',
        'SecureResponseFiles': 'true',
        'SurveyExpiration': None,
        'SurveyTermination': 'DefaultMessage',
        'Header': '',
        'Footer': '',
        'ProgressBarDisplay': 'None',
        'PartialData': '+3 days',
        'PreviousButton': ' ← ',
        'NextButton': ' → ',
        'SkinLibrary': 'nyu',
        'SkinType': 'templated',
        'Skin': {
            'brandingId': None,
            'templateId': '*base',
            'overrides': {
                'contrast': 0.3,
                'questionsContainer': {
                    'on': True}}},
        'NewScoring': 1,
        'CustomStyles': [],
        'QuestionsPerPage': '1',
        'PageTransition': 'fade',
        'EOSMessage': '',
        'ShowExportTags': 'false',
        'CollectGeoLocation': 'false',
        'SurveyTitle': 'Online Survey Software | Qualtrics Survey Solutions',
        'SurveyMetaDescription': 'Qualtrics sophisticated online survey software solutions make creating online surveys easy. Learn more about Research Suite and get a free account today.',
        'PasswordProtection': 'No',
        'AnonymizeResponse': 'No',
        'Password': '',
        'RefererCheck': 'No',
        'RefererURL': 'http://',
        'UseCustomSurveyLinkCompletedMessage': None,
        'SurveyLinkCompletedMessage': '',
        'SurveyLinkCompletedMessageLibrary': '',
        'ResponseSummary': 'No',
        'EOSMessageLibrary': '',
        'EmailThankYou': 'false',
        'ThankYouEmailMessageLibrary': None,
        'ThankYouEmailMessage': None,
        'ValidateMessage': 'false',
        'ValidationMessageLibrary': None,
        'InactiveSurvey': 'DefaultMessage',
        'PartialDataCloseAfter': 'LastActivity',
        'ActiveResponseSet': None,
        'InactiveMessageLibrary': '',
        'InactiveMessage': '',
        # 'AvailableLanguages': {
        #    'EN': []},
        # 'SurveyLanguage': 'EN',
        'SurveyStartDate': None,
        'SurveyExpirationDate': None})

    update_options(SurveyOptions, SurveyID, apiToken=apiToken, dataCenter=dataCenter)
