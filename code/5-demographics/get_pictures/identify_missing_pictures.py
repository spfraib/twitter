import gzip
import argparse
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--tar_type", type=str, help='Whether to look at the tar or resized_tars')
    args = parser.parse_args()
    return args


args = get_args_from_command_line()
dir = os.path.join("/scratch/spf248/twitter/data/classification", args.country_code, "profile_pictures_sam")

def save_users_with_picture():
    files = os.listdir(dir)
    known_users = []
    for file in files:
        path = os.path.join(dir, file)
        known_users += [f.split('.')[0] for f in os.listdir(path)]

    known_users = set(known_users)
    with open(f'known_users_{country_code}.pkl', 'wb') as file:
        pickle.dump(known_users, file)

def load_all_users():
    all_users = []
    dir_name = f"/scratch/spf248/twitter/data/classification/{country_code}/users/"
    # filenames = os.listdir("/scratch/spf248/twitter/data/classification/US/users/")
    all_users_paths = list(glob(os.path.join(dir_name, '*.parquet')))
    for file_name in all_users_paths:
        users = pq.read_table(file_name, columns=['user_id']).to_pandas()
        all_users += list(users)
    all_users = set(all_users)
    return all_users

def load_missing_users():
    if not os.path.isfile(f'known_users_{country_code}.pkl'):
        save_known_users()
    with open('known_users.pkl', 'rb') as file:
        known_users = pickle.load(file)
    all_users = load_all_users()
    missing_users = all_users - known_users
    return missing_users

missing_users = load_missing_users()
with open(f'missing_users_{country_code}.pkl', 'wb') as file:
    pickle.dump(missing_users, file)
if __name__ == '__main__':
