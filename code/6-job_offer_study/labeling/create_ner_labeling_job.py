import boto3

keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/keys/mturk'
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
    LabelingJobName='example-ner-labeling-job',
    LabelAttributeName='label',
    InputConfig={
        'DataSource': {
            'S3DataSource': {
                'ManifestS3Uri': 's3://bucket/path/manifest-with-input-data.json'
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
        'S3OutputPath': 's3://bucket/path/file-to-store-output-data',
        'KmsKeyId': 'string',
        'SnsTopicArn': 'string'
    },
    RoleArn='arn:aws:iam::*:role/*',
    LabelCategoryConfigS3Uri='s3://bucket/path/label-categories.json',
    StoppingConditions={
        'MaxHumanLabeledObjectCount': 123,
        'MaxPercentageOfInputDatasetLabeled': 123
    },
    HumanTaskConfig={
        'WorkteamArn': 'arn:aws:sagemaker:region:*:workteam/private-crowd/*',
        'UiConfig': {
            'UiTemplateS3Uri': 's3://bucket/path/worker-task-template.html',
            'HumanTaskUiArn': 'string'
        },
        'PreHumanTaskLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:PRE-NamedEntityRecognition',
        'TaskKeywords': [
            'Named Entity Recognition',
        ],
        'TaskTitle': 'Named entity Recognition task',
        'TaskDescription': 'Apply the labels provided to specific words or phrases within the larger text block.',
        'NumberOfHumanWorkersPerDataObject': 123,
        'TaskTimeLimitInSeconds': 123,
        'TaskAvailabilityLifetimeInSeconds': 123,
        'MaxConcurrentTaskCount': 123,
        'AnnotationConsolidationConfig': {
            'AnnotationConsolidationLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:ACS-NamedEntityRecognition'
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