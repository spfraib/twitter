import boto3
import os
import argparse


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--HITId", type=str)
    parser.add_argument("--worker_id", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args_from_command_line()

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

    mturk.approve_assignment(AssignmentId=assignment_worker_ids_dict[args.worker_id], RequesterFeedback='',
                             OverrideRejection=True)


if __name__ == '__main__':
    main()
