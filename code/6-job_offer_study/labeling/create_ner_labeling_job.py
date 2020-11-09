import boto3

keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/mturk/keys'
with open(os.path.join(keys_path, 'access_key_id.txt'), 'r') as f:
    access_key_id = f.readline().strip()

with open(os.path.join(keys_path, 'secret_access_key.txt'), 'r') as f:
    secret_access_key = f.readline().strip()

sagemaker = boto3.client('sagemaker',
                     aws_access_key_id=access_key_id,
                     aws_secret_access_key=secret_access_key,
                     region_name='us-east-1',
                     #endpoint_url='https://mturk-requester.us-east-1.amazonaws.com'
                     )

response = sagemaker.create_labeling_job(
    LabelingJobName='string',
    LabelAttributeName='string',
    InputConfig={
        'DataSource': {
            'S3DataSource': {
                'ManifestS3Uri': 'string'
            },
            'SnsDataSource': {
                'SnsTopicArn': 'string'
            }
        },
        'DataAttributes': {
            'ContentClassifiers': [
                'FreeOfPersonallyIdentifiableInformation'|'FreeOfAdultContent',
            ]
        }
    },
    OutputConfig={
        'S3OutputPath': 'string',
        'KmsKeyId': 'string',
        'SnsTopicArn': 'string'
    },
    RoleArn='string',
    LabelCategoryConfigS3Uri='string',
    StoppingConditions={
        'MaxHumanLabeledObjectCount': 123,
        'MaxPercentageOfInputDatasetLabeled': 123
    },
    LabelingJobAlgorithmsConfig={
        'LabelingJobAlgorithmSpecificationArn': 'string',
        'InitialActiveLearningModelArn': 'string',
        'LabelingJobResourceConfig': {
            'VolumeKmsKeyId': 'string'
        }
    },
    HumanTaskConfig={
        'WorkteamArn': 'string',
        'UiConfig': {
            'UiTemplateS3Uri': 'string',
            'HumanTaskUiArn': 'string'
        },
        'PreHumanTaskLambdaArn': 'string',
        'TaskKeywords': [
            'string',
        ],
        'TaskTitle': 'string',
        'TaskDescription': 'string',
        'NumberOfHumanWorkersPerDataObject': 123,
        'TaskTimeLimitInSeconds': 123,
        'TaskAvailabilityLifetimeInSeconds': 123,
        'MaxConcurrentTaskCount': 123,
        'AnnotationConsolidationConfig': {
            'AnnotationConsolidationLambdaArn': 'string'
        },
        'PublicWorkforceTaskPrice': {
            'AmountInUsd': {
                'Dollars': 123,
                'Cents': 123,
                'TenthFractionsOfACent': 123
            }
        }
    },
    Tags=[
        {
            'Key': 'string',
            'Value': 'string'
        },
    ]
)