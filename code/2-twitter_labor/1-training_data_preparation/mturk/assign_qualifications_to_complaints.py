import boto3
import math
import argparse
import os

keys_path = '/scratch/mt4493/twitter_labor/twitter-labor-data/data/keys/mturk'
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

worker_id_complaint_dic = {
    'BR': ['A3GZZOG8SFEGRE',
           'A2EXET9P3UV1BE',
           'A20AFZHTCOHL2A',
           'A2519LCLPN7Y6I',
           'A1MXDBILWV8RXN',
           'A3TU0NGBDKSGLI',
           'A3B0TDW8S89NSI'],
    'US': ['A2WWYVKGZZXBOB',
           'A3POD149IG0DIW',
           'A1NRWXWPCAXDYP',
           'A5YVMQWA2IYPF',
           'A317MV1MRFSNZB',
           'A28AXX4NCWPH1F',
           'A1PHDT66U6IK4Q',
           'AFUUPNBIKHRFZ',
           'AD20ZSEI47B13',
           'A2ASTDMWBCFIP0',
           'A37XJVQF62ZYC',
           'A2NZ7RMSBXESNI',
           'A3FY6THWKRYN9M',
           'A1ZAG3PSITDO8W',
           'AE1YA7Q3UKPRI',
           'AROUPPNOA783R',
           'AVJUIF9QHQRY8'],
    'MX': ['A3JWCEGDOK8IXH',
           'A33DJL9BYFCNDB']
}

qualification_type_id = '3N54K5T7LDDGBF1H17J77WCI6Q9YZP'

for country_code in ['US', 'MX', 'BR']:
    for worker_id in worker_id_complaint_dic[country_code]:
        try:
            mturk.associate_qualification_with_worker(
            QualificationTypeId=qualification_type_id,
            WorkerId=worker_id,
            IntegerValue=1,
            SendNotification=False)
        except:
            print(country_code)
            print(worker_id)