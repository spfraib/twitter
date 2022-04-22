import tarfile
import argparse
import logging
from pathlib import Path
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import tempfile
from glob import glob
from m3inference import M3Inference
import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import ast


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default=None)
    return parser.parse_args()


def get_env_var(varname, default):
    if os.environ.get(varname) is not None:
        var = int(os.environ.get(varname))
        logger.info(f'{varname}: {var}')
    else:
        var = default
        logger.info(f'{varname}: {var}, (Default)')
    return var


def resize_img(img_path, img_out_path, filter=Image.BILINEAR, force=False, url=None):
    try:
        img = Image.open(img_path).convert("RGB")
        if img.size[0] + img.size[1] < 400 and not force:
            logger.info(f'{img_path} / {url} is too small. Skip.')
            os.remove(img_path)
            return
        img = img.resize((224, 224), filter)
        img.save(img_out_path)
        os.remove(img_path)
    except Exception as e:
        logger.warning(f'Error when resizing {img_path} / {url}\nThe error message is {e}\n')


def resize_imgs(src_root, dest_root, src_list=None, filter=Image.BILINEAR, force=False):
    if not os.path.exists(src_root):
        raise FileNotFoundError(f"{src_root} does not exist.")

    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    src_list = glob(os.path.join(src_root, '*')) if src_list is None else src_list

    des_set = set([os.path.relpath(img_path, dest_root).replace('.jpeg', '')
                   for img_path in glob(os.path.join(dest_root, '*'))])

    for img_path in tqdm(src_list, desc='resizing images', disable=logging.root.level >= logging.WARN):

        img_name = os.path.splitext(os.path.relpath(img_path, src_root))[0]
        if not force and img_name in des_set:
            logger.debug(f"{img_name} exists. Skipping...")
            continue
        else:
            out_path = os.path.join(dest_root, img_name) + '.jpeg'
            logger.debug(f'{img_name} not found in {dest_root}. Resizing to {out_path}')
            resize_img(img_path, out_path, filter=filter, force=force)


def set_lang(country_code):
    languages = {'US': 'en', 'MX': 'es', 'BR': 'pt', 'NG': 'en'}
    return languages[country_code]


def extract(row, tmpdir, tar_dir):
    # user = row['id']
    # if user not in mapping_dict:
    #     return None
    # else:
    tfilename = row['tfilename']
    tmember = row['tmember']
    if not pd.isna(tfilename) and not pd.isna(tmember):
        os.makedirs(f'{tmpdir}/original_pics', exist_ok=True)
        with tarfile.open(os.path.join(tar_dir, tfilename), mode='r', ignore_zeros=True) as tarf:
            for member in tarf.getmembers():
                if member.name == tmember:
                    tmember = member
                    tmember.name = os.path.basename(member.name)
                    break
            tarf.extract(tmember, path=f'{tmpdir}/original_pics')
        return os.path.join(tmpdir, 'original_pics', tmember.name)
    else:
        return None


def get_resized_path(orig_img_path, src_root, dest_root):
    img_name = os.path.splitext(os.path.relpath(orig_img_path, src_root))[0]
    out_path = os.path.join(dest_root, img_name) + '.jpeg'
    if os.path.exists(out_path):
        return out_path
    else:
        return None

def findMax(pred):
    label = None
    maxm = 0
    for key,val in pred.items():
        if val>maxm:
            maxm = val
            label = key
    return label

def save_dict_to_json(data_dict, outfile):
    with open(outfile, 'a') as file:
        file.write('{}\n'.format(json.dumps(data_dict)))

def retrieve_known_ids(output_dir):
    json_paths_list = Path(output_dir).glob('*.json')
    known_ids_list = list()
    for json_path in json_paths_list:
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    user_dict = ast.literal_eval(line)
                    known_ids_list.append(user_dict['user'])
                except:
                    continue
    return known_ids_list

def retrieve_non_resizable_ids(err_dir):
    json_paths_list = Path(err_dir).glob('*.json')
    not_resizable_id_list = list()
    for json_path in json_paths_list:
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    line = line.replace('\n', '')
                    not_resizable_id_list.append(str(line))
                except:
                    continue
    return not_resizable_id_list

if __name__ == '__main__':
    args = get_args_from_command_line()
    # logger.info(f'Country code: {args.country_code}')
    # define env vars
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    # define paths and paths to be treated
    tar_dir = f'/scratch/spf248/twitter_data_collection/data/demographics/profile_pictures/tars'
    user_dir = f'/scratch/spf248/twitter_data_collection/data/user_timeline/profiles_with_tar_path'
    # user_mapping_path = f'/scratch/spf248/twitter_data_collection/data/demographics/profile_pictures/tars/user_map_dict_all.json'
    output_dir = f'/scratch/spf248/twitter_data_collection/data/demographics/inference_results'
    err_dir = os.path.join(output_dir, 'err')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # store inferences already done in known_ids
    files_on_output = glob(os.path.join(output_dir, "*"))
    if len(files_on_output) > 0:
        known_ids = set(retrieve_known_ids(output_dir=output_dir))
        logger.info(f"A total of {len(known_ids)} ids were already known.")
    else:
        known_ids = set([])
    # load user mapping
    # with open(user_mapping_path, 'r') as fp:
    #     user_image_mapping_dict = json.load(fp)
    # logger.info('Loaded the user image mapping')
    # select users and load data
    # selected_users_list = list(
    #     np.array_split(glob(os.path.join(user_dir, '*.parquet')), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    selected_users_list = list(
        np.array_split(glob(os.path.join(user_dir, '*.parquet')), 2000)[0])
    logger.info(f'# retained files: {len(selected_users_list)}')
    selected_users_list = [selected_users_list[0]]
    logger.info(selected_users_list)
    if len(selected_users_list) > 0:
        df_list = list()
        for parquet_path in selected_users_list:
            df = pd.read_parquet(parquet_path)
            if args.country_code != 'all':
                df = df.loc[df['country_short'] == args.country_code]
            df_list.append(df)
        df = pd.concat(df_list).reset_index(drop=True)
        del df_list
        logger.info(f'Initial df size: {df.shape[0]}')
        print(df.loc[df['tfilename'].isnull()].shape[0])
        df = df[~df["tfilename"].isnull()]
        df = df[~df["tmember"].isnull()]
        logger.info(f'df size after keeping users with image: {df.shape[0]}')
        df = df[~df["user_id"].isin(known_ids)]
        logger.info(f'df size after dropping known ids: {df.shape[0]}')
        if os.path.exists(err_dir):
            total_not_resizable_id_list = retrieve_non_resizable_ids(err_dir=err_dir)
            df = df[~df["user_id"].isin(total_not_resizable_id_list)]
            logger.info(f'df size after dropping users with non resizable pictures: {df.shape[0]}')
        df = df.rename(columns={'user_id': 'id', 'user_profile_image_url_https': 'img_path', })
        df = df[['id', 'user_name', 'user_screen_name', 'user_description', 'img_path', 'tfilename', 'tmember']]
        df['lang'] = set_lang(country_code=args.country_code)
        for (ichunk, chunk) in enumerate(np.array_split(df, 10)):
            if chunk.shape[0] == 0:
                continue
            initial_chunk_shape = chunk.shape[0]
            chunk = [chunk[0]]
            logger.info(f'Starting with chunk {ichunk}. Chunk size is {initial_chunk_shape} users.')
            with tempfile.TemporaryDirectory() as tmpdir:
                logger.info('Extract pictures from tars')
                chunk['original_img_path'] = chunk.apply(
                    lambda x: extract(row=x, tmpdir=tmpdir, tar_dir=tar_dir), axis=1)
                chunk = chunk.loc[~chunk['original_img_path'].isnull()].reset_index(drop=True)
                logger.info(f'Chunk shape with pictures: {chunk.shape[0]}')
                logger.info('Resize imgs')
                resize_imgs(src_root=f'{tmpdir}/original_pics', dest_root=f'{tmpdir}/resized_pics')
                chunk['img_path'] = chunk['original_img_path'].apply(
                    lambda x: get_resized_path(orig_img_path=x, src_root=f'{tmpdir}/original_pics',
                                               dest_root=f'{tmpdir}/resized_pics'))
                not_resizable_chunk = chunk.loc[chunk['img_path'].isnull()].reset_index(drop=True)
                if not_resizable_chunk.shape[0] > 0:
                    not_resizable_id_list = not_resizable_chunk['id'].tolist()
                else:
                    not_resizable_id_list = list()
                chunk = chunk[['id', 'user_name', 'user_screen_name', 'user_description', 'lang', 'img_path']]
                initial_chunk_shape = chunk.shape[0]
                chunk = chunk.loc[~chunk['img_path'].isnull()].reset_index(drop=True)
                logger.info(f'Chunk size with resized pics: {chunk.shape[0]}')
                chunk = chunk.dropna()
                df_json = json.loads(chunk.to_json(orient='records'))
                # Run inference
                logger.info('Launching inference')
                m3 = M3Inference()
                predictions = m3.infer(df_json)
                # Save inference output
                if predictions:
                    logger.info('Saving inference output')
                    json_output_path = os.path.join(output_dir, f"{SLURM_JOB_ID}.json")
                    rows = []
                    for item in predictions.items():
                        user, pred = item
                        row_dict = {"user": user, "gender": findMax(pred['gender']),
                               "age": findMax(pred['age']), "org": findMax(pred['org'])}
                        save_dict_to_json(data_dict=row_dict, outfile=json_output_path)
                # Save errors
                if len(not_resizable_id_list) > 0:
                    logger.info('Saving errors')
                    err_dir = os.path.join(output_dir, 'err')
                    if not os.path.exists(err_dir):
                        os.makedirs(err_dir, exist_ok=True)
                    outfile = os.path.join(err_dir, f"{SLURM_JOB_ID}.json")
                    for error_id in not_resizable_id_list:
                        with open(outfile, 'a') as file:
                            file.write(f'{error_id}\n')
