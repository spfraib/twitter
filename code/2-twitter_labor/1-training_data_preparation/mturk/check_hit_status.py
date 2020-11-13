import boto3
import os
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--HITId", type=str)

    args = parser.parse_args()
    return args

def main():
    # Get args
    args = get_args_from_command_line()
    # Load mturk client
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

    # Get hit status and print
    response = mturk.get_hit(HITId=args.HITId)
    for key, value in response['HIT'].items():
        print(key, ' : ', value)
    
if __name__ == '__main__':
    main()