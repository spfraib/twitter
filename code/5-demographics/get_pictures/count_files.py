import os
import argparse
def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    args = parser.parse_args()
    return args

args = get_args_from_command_line()

dir = f"/scratch/spf248/twitter/data/classification/{args.country_code}/profile_pictures_sam/"

files = os.listdir(dir)
count = 0

for file in files:
        path = os.path.join(dir,file)
        count += len(os.listdir(path))

print(count)
