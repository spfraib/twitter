import boto3
import math
import argparse
import os


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="US")
    parser.add_argument("--n_workers", type=int, help="number of workers",
                        default=20)
    parser.add_argument("--survey_link", type=str)
    parser.add_argument("--block_size", help='number of tweets per worker', type=int)
    parser.add_argument("--version_number", type=str)
    parser.add_argument("--mode", type=str, help='Whether to create HIT in sandbox or in production')
    parser.add_argument("--language_qualification", type=int, help='')

    args = parser.parse_args()
    return args


def question_generator(country_code, survey_link, instructions_dict, survey_link_text_dict, worker_input_text_dict,
                       submit_dict):
    xml_wrapper_begin = """
    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
    <HTMLContent><![CDATA[
    <!-- YOUR HTML BEGINS -->
    <!DOCTYPE html>
    """
    xml_wrapper_end = """
    <!-- YOUR HTML ENDS -->
    ]]>
    </HTMLContent>
    <FrameHeight>0</FrameHeight>
    </HTMLQuestion>
    """

    with open(
            "/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/1-training_data_preparation/mturk/template.html",
            "r") as f:
        content = f.read()

    content = content.replace("${INSTRUCTIONS}", instructions_dict[country_code])
    content = content.replace("${SURVEY_LINK}", survey_link)
    content = content.replace("${SURVEY_LINK_TEXT}", survey_link_text_dict[country_code])
    content = content.replace("${WORKER_INPUT_TEXT}", worker_input_text_dict[country_code])
    content = content.replace("${SUBMIT}", submit_dict[country_code])

    return xml_wrapper_begin + content + xml_wrapper_end


args = get_args_from_command_line()
print('Args:', args)

ntweets = args.block_size
time_to_complete = int(math.ceil((3 / 5) * ntweets))

money_per_tweet_dict = {
    'US': 0.1,
    'BR': 0.06,
    'MX': 0.1
}

money_for_hit = int(money_per_tweet_dict[args.country_code] * ntweets)

keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/keys/mturk'
with open(os.path.join(keys_path, 'access_key_id.txt'), 'r') as f:
    access_key_id = f.readline().strip()

with open(os.path.join(keys_path, 'secret_access_key.txt'), 'r') as f:
    secret_access_key = f.readline().strip()

# MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

if args.mode == 'production':
    create_hits_in_production = True
elif args.mode == 'sandbox':
    create_hits_in_production = False

environments = {
    "production": {
        "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
        "preview": "https://www.mturk.com/mturk/preview"
    },
    "sandbox": {
        "endpoint":
            "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
        "preview": "https://workersandbox.mturk.com/mturk/preview"
    },
}
mturk_environment = environments["production"] if create_hits_in_production else environments["sandbox"]

mturk = boto3.client('mturk',
                     aws_access_key_id=access_key_id,
                     aws_secret_access_key=secret_access_key,
                     region_name='us-east-1',
                     endpoint_url=mturk_environment['endpoint']
                     )

title_dict = {
    'US': 'Read %d English Tweets and answer a few questions' % (ntweets),
    'MX': 'Lea %d Tweets en español mexicano y responda algunas preguntas' % (ntweets),
    'BR': 'Leia %d tweets em português e responda algumas perguntas' % (ntweets)
}

description_dict = {
    'US': 'Assess the employment status of Twitter users',
    'MX': 'Evaluar la situación laboral de usuarios de Twitter',
    'BR': 'Avalie a situação de emprego de usuários do Twitter'
}

keywords_dict = {
    'US': 'Survey, Labeling, Twitter',
    'MX': 'Encuesta, Etiquetaje/labelling, Twitter',
    'BR': 'Pesquisa, rotulagem, Twitter'
}

instructions_dict = {
    'US': 'We would like to invite you to participate in a web-based survey. The survey contains a list of <b>%d</b> questions. It should take you approximately <b>%d</b> minutes and you will be paid <b>%.2f</b> USD for successful completion of the survey. A valid worker ID is needed to receive compensation. At the end of the survey you will receive a completion code, which you need to enter below in order to receive payment. You may start the survey by clicking the survey link below.' % (
        ntweets, time_to_complete, money_for_hit),
    'MX': 'Nos gustaría invitarle a participar en una encuesta en la web. La encuesta contiene una lista de preguntas de <b>%d</b> preguntas. Le tomará aproximadamente <b>%d</b> minutos y se le pagará <b>%.2f</b> USD por completar la encuesta exitosamente. Se necesita una identificación de trabajador válida para recibir la compensación. Al final de la encuesta recibirá un código de finalización, que deberá introducir a continuación para recibir el pago. Puede comenzar la encuesta haciendo clic en el enlace de la encuesta que aparece a continuación.' % (
        ntweets, time_to_complete, money_for_hit),
    'BR': 'Gostaríamos de te convidar a participar de uma pesquisa online. A pesquisa contém uma lista de <b>%d</b> perguntas. Esta pesquisa deve demorar aproximadamente <b>%d</b> minutos e você receberá <b>%.2f</b> USD pela conclusão do seu trabalho. É necessária uma identificação de trabalhador válida para receber a compensação. Ao final da pesquisa, você receberá um código de preenchimento, que deverá ser inserido abaixo para receber o pagamento. Você pode iniciar a pesquisa clicando no link abaixo.' % (
        ntweets, time_to_complete, money_for_hit)
}

survey_link_text_dict = {
    'US': 'Survey link',
    'MX': 'Enlace de la encuesta',
    'BR': 'Link para pesquisa'
}

worker_input_text_dict = {
    'US': 'Provide the survey code here:',
    'MX': 'Proporcione el código de la encuesta aquí:',
    'BR': 'Entre com o código da pesquisa aqui:'
}

submit_dict = {
    'US': 'Submit',
    'MX': 'Enviar',
    'BR': 'Enviar'
}
question = question_generator(country_code=args.country_code, survey_link=args.survey_link,
                              instructions_dict=instructions_dict, survey_link_text_dict=survey_link_text_dict,
                              worker_input_text_dict=worker_input_text_dict, submit_dict=submit_dict)
print("QUESTION:", question)

if args.country_code == 'MX':
    QualificationRequirements_list = [
        {
            'QualificationTypeId': '3RDXJZR9A1H33MQ79TZZWYBXX8WCYD',
            'Comparator': 'DoesNotExist',
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        },
        {
            'QualificationTypeId': '00000000000000000071',  # Worker_Locale
            'Comparator': 'In',
            'LocaleValues': [{
                                # 'Country': 'US'}, {
                                 'Country': 'MX'}, {
                                 'Country': 'CO'}, {
                                 'Country': 'ES'}, {
                                 'Country': 'AR'}, {
                                 'Country': 'PE'}, {
                                 'Country': 'VE'}, {
                                 'Country': 'CL'}, {
                                 'Country': 'EC'}, {
                                 'Country': 'GT'}, {
                                 'Country': 'CU'}, {
                                 'Country': 'BO'}, {
                                 'Country': 'DO'}, {
                                 'Country': 'HN'}, {
                                 'Country': 'PY'}, {
                                 'Country': 'SV'}, {
                                 'Country': 'NI'}, {
                                 'Country': 'CR'}, {
                                 'Country': 'PA'}, {
                                 'Country': 'UY'}, {
                                 'Country': 'GQ'}, {
                                 'Country': 'GI'}],
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        }
        # Qualification: speaks Spanish
        # {
        #     'QualificationTypeId': '3O3C9VE8V1CPLASHO374P1HIP94SH7',
        #     'Comparator': 'EqualTo',
        #     'IntegerValues': [1],
        #     'RequiredToPreview': True,
        #     'ActionsGuarded': 'DiscoverPreviewAndAccept'}
    ]

# elif args.country_code == 'US':
#     QualificationRequirements_list = [
#         {
#             'QualificationTypeId': '00000000000000000071',  # Worker_Locale
#             'Comparator': 'EqualTo',
#             'LocaleValues': [{
#                 'Country': args.country_code}],
#             'RequiredToPreview': True,
#             'ActionsGuarded': 'PreviewAndAccept'
#         },
#         {
#             'QualificationTypeId': '3YLTB9JB8TED72KIAHT6K4NASKY63F',
#             'Comparator': 'DoesNotExist',
#             'RequiredToPreview': True,
#             'ActionsGuarded': 'PreviewAndAccept'
#         },
#         {
#             'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
#             'Comparator': 'Exists',
#             'RequiredToPreview': True,
#             'ActionsGuarded': 'PreviewAndAccept'
#         }
#     ]

else:
    QualificationRequirements_list = [
        {
            'QualificationTypeId': '00000000000000000071',  # Worker_Locale
            'Comparator': 'EqualTo',
            'LocaleValues': [{
                'Country': args.country_code}],
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        },
        {
            #'QualificationTypeId': '3YLTB9JB8TED72KIAHT6K4NASKY63F',
            'QualificationTypeId': '3RDXJZR9A1H33MQ79TZZWYBXX8WCYD',
            'Comparator': 'DoesNotExist',
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        }
    ]
new_hit = mturk.create_hit(
    MaxAssignments=args.n_workers,
    AutoApprovalDelayInSeconds=172800,
    LifetimeInSeconds=259200,
    AssignmentDurationInSeconds=10800,
    Reward=str(money_for_hit),
    Title=f'{title_dict[args.country_code]} v{args.version_number}',
    Description=description_dict[args.country_code],
    Keywords=keywords_dict[args.country_code],
    QualificationRequirements=QualificationRequirements_list if create_hits_in_production else list(),
    Question=question
)

print("A new HIT has been created. You can preview it here:")
print(f"{mturk_environment['preview']}?groupId={new_hit['HIT']['HITGroupId']}")
print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=
