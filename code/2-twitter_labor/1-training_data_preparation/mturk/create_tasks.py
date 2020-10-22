import boto3
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
    parser.add_argument("--version_number", type=str)

    args = parser.parse_args()
    return args


def question_generator(country_code, survey_link, instructions_dict, survey_link_text_dict, worker_input_text_dict):
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

    content.replace("${INSTRUCTIONS}", instructions_dict[country_code])
    content.replace("${SURVEY_LINK}", survey_link)
    content.replace("${SURVEY_LINK_TEXT}", survey_link_text_dict[country_code])
    content.replace("${WORKER_INPUT_TEXT}", worker_input_text_dict[country_code])

    return xml_wrapper_begin + content + xml_wrapper_end


args = get_args_from_command_line()
keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/mturk/keys'
with open(os.path.join(keys_path, 'access_key_id.txt'), 'r') as f:
    access_key_id = f.readline().strip()

with open(os.path.join(keys_path,'secret_access_key.txt'), 'r') as f:
    secret_access_key = f.readline().strip()

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
mturk = boto3.client('mturk',
                     aws_access_key_id=access_key_id,
                     aws_secret_access_key=secret_access_key,
                     region_name='us-east-1',
                     endpoint_url=MTURK_SANDBOX
                     )

title_dict = {
    'US': 'Read 50 English Tweets and answer a few questions',
    'MX': 'Lea 50 Tweets en español y responda algunas preguntas',
    'BR': 'Leia 50 tweets em português e responda algumas perguntas'
}

description_dict = {
    'US': 'Assess the employment status of Twitter users',
    'MX': 'Evaluar la situación laboral de los usuarios de Twitter',
    'BR': 'Avalie a situação de emprego de usuários do Twitter'
}

keywords_dict = {
    'US': 'Survey, Labeling, Twitter',
    'MX': 'Encuesta, Etiquetaje/labelling, Twitter',
    'BR': 'Pesquisa, rotulagem, Twitter'
}

instructions_dict = {
    'US': 'We would like to invite you to participate in a web-based survey. The survey contains a list of <chunksize, default=50> questions. It should take you approximately 25 minutes and you will be paid <reward, default=4USD> for successful completion of the survey. A valid worker ID is needed to receive compensation. At the end of the survey you will receive a completion code, which you need to enter below in order to receive payment. You may start the survey by clicking the survey link below.',
    'MX': 'Nos gustaría invitarle a participar en una encuesta en la web. La encuesta contiene una lista de preguntas de <chunksize, default = 50> preguntas. Le tomará aproximadamente 25 minutos y se le pagará <reward, default=4USD> por completar la encuesta exitosamente. Se necesita una identificación de trabajador válida para recibir la compensación. Al final de la encuesta recibirá un código de finalización, que deberá introducir a continuación para recibir el pago. Puede comenzar la encuesta haciendo clic en el enlace de la encuesta que aparece a continuación.',
    'BR': 'Gostaríamos de te convidar a participar de uma pesquisa online. A pesquisa contém uma lista de <chunksize, default = 50> perguntas. Esta pesquisa deve demorar aproximadamente 25 minutos e você receberá <recompensa, padrão = 4USD> pela conclusão do seu trabalho. É necessária uma identificação de trabalhador válida para receber a compensação. Ao final da pesquisa, você receberá um código de preenchimento, que deverá ser inserido abaixo para receber o pagamento. Você pode iniciar a pesquisa clicando no link abaixo.'
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
question = question_generator(country_code=args.country_code, survey_link=args.survey_link,
                              instructions_dict=instructions_dict, survey_link_text_dict=survey_link_text_dict,
                              worker_input_text_dict=worker_input_text_dict)
print("QUESTION:", question)

new_hit = mturk.create_hit(
    MaxAssignments=args.n_workers,
    AutoApprovalDelayInSeconds=86400,
    LifetimeInSeconds=259200,
    AssignmentDurationInSeconds=10800,
    Reward='4',
    Title=f'{title_dict[args.country_code]} v{args.version_number}',
    Description=description_dict[args.country_code],
    Keywords=keywords_dict[args.country_code],
    QualificationRequirements=[
        {
            'QualificationTypeId': '00000000000000000071',  # Worker_Locale
            'Comparator': 'EqualTo',
            'LocaleValues': [{
                                 'Country': args.country_code}],
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
        },
        {
            'QualificationTypeId': '3YLTB9JB8TED72KIAHT6K4NASKY63F',
            'Comparator': 'DoesNotExist',
            'RequiredToPreview': True,
            'ActionsGuarded': 'PreviewAndAccept'
            }
    ],
    Question=question
)

print("A new HIT has been created. You can preview it here:")
print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=
