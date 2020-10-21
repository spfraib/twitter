import boto3
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        help="Country code",
                        default="US")
    parser.add_argument("--n_workers", type=int, help="number of workers",
                        default=20)
    args = parser.parse_args()
    return args

args = get_args_from_command_line()
MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

with open('/scratch/mt4493/twitter_labor/twitter-labor-data/data/mturk/keys/access_key_id.txt', 'r') as f:
    access_key_id = f.readline()

with open('/scratch/mt4493/twitter_labor/twitter-labor-data/data/mturk/keys/secret_access_key.txt', 'r') as f:
    secret_access_key = f.readline()

mturk = boto3.client('mturk',
                     aws_access_key_id=access_key_id,
                     aws_secret_access_key=secret_access_key,
                     region_name='us-east-1',
                     endpoint_url=MTURK_SANDBOX
                     )

question = open(name='questions.xml', mode='r').read()

Title_dict = {
    'US': 'Read 50 English Tweets and answer a few questions',
    'MX': 'Lea 50 Tweets en español y responda algunas preguntas',
    'BR': ''
}

Description_dict = {
    'US': 'Assess the employment status of Twitter users',
    'MX': 'Evaluar la situación laboral de los usuarios de Twitter',
    'BR': ''
}

Keywords_dict = {
    'US': 'Survey, Labeling, Twitter',
    'MX': 'Encuesta, Etiquetaje/labelling, Twitter',
    'BR': ''
}

new_hit = mturk.create_hit(
    MaxAssignments= args.n_workers,
    AutoApprovalDelayInSeconds=0,
    LifetimeInSeconds=259200,
    AssignmentDurationInSeconds=10800,
    Reward='4',
    Title= Title_dict[args.country_code],
    Description= Description_dict[args.country_code],
    Keywords=Keywords_dict[args.country_code],
    QualificationRequirements=[
        {'QualificationTypeId': '00000000000000000071', #Worker_Locale
         'Comparator': 'EqualTo',
         'LocaleValues': [
             {'Country': 'US'},],
         'RequiredToPreview': True
         },
        #{'QualificationTypeId': '',
        # 'Comparator': '',
        # 'XValues': []
        #}
    ],
    HITLayoutID='3D31OFTG75V3UNIZ5K2IBUGE1XJIUU',
    HITLayoutParameters= [
        {
            
        }
    ]
)


print("A new HIT has been created. You can preview it here:")
print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=

