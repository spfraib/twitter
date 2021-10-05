import tarfile
import argparse
import logging
from pathlib import Path
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import tempfile

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--mode", type=str)
    return parser.parse_args()


def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        logger.info(f'{varname}: {var}')
    else:
        var = default
        logger.info(f'{varname}: {var}, (Default)')
    return var


def sh(script):
    os.system("bash -c '%s'" % script)


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

    src_list = glob.glob(os.path.join(src_root, '*')) if src_list is None else src_list

    des_set = set([os.path.relpath(img_path, dest_root).replace('.jpeg', '')
                   for img_path in glob.glob(os.path.join(dest_root, '*'))])

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
    languages = {'US': 'en', 'MX': 'es', 'BR': 'pt'}
    return languages[country_code]


def extract(row, tmpdir, user_image_mapping_dict):
    user = row['id']
    if user not in user_image_mapping_dict:
        return np.nan
    else:
        tfilename, tmember = user_image_mapping[user]
        os.makedirs(f'{tmpdir}/original_pics', exist_ok=True)
        with tarfile.open(tfilename, mode='r', ignore_zeros=True) as tarf:
            for member in tarf.getmembers():
                if member.name == tmember:
                    tmember = member
                    tmember.name = os.path.basename(member.name)
                break
            tarf.extract(tmember, path=f'{tmpdir}/original_pics')
        return os.path.join(tmpdir, 'original_pics', tmember.name)


def get_resized_path(orig_img_path, src_root, dest_root):
    img_name = os.path.splitext(os.path.relpath(orig_img_path, src_root))[0]
    out_path = os.path.join(dest_root, img_name) + '.jpeg'
    if os.path.exists(out_path):
        return out_path
    else:
        return None


if __name__ == '__main__':
    args = get_args_from_command_line()
    # define env vars
    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
    SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())
    # define paths and paths to be treated
    user_dir = f'/scratch/spf248/twitter/data/user_timeline/user_timeline_crawled/{args.country_code}'
    user_mapping_path = f'/scratch/spf248/twitter/data/demographics/profile_pictures/tars/user_map_dict_{args.country_code}.json'
    # load user mapping
    with open(user_mapping_path, 'r') as fp:
        user_image_mapping_dict = json.load(fp)
    # select users and load data
    selected_users_list = list(
        np.array_split(glob(os.path.join(user_dir, '*.parquet')), SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
    logger.info(f'# retained files: {len(selected_users_list)}')
    if len(selected_users_list) > 0:
        df = pq.read_table(source=selected_parquets).to_pandas()
        df = df.rename(columns={'user_id': 'id', 'profile_image_url_https': 'img_path'})
        df = df[['id', 'name', 'screen_name', 'description', 'lang', 'img_path']]
        df['lang'] = set_lang(country_code=args.country_code)
        for (ichunk, chunk) in enumerate(np.array_split(df, 10)):
            if chunk.shape[0] == 0:
                continue
            with tempfile.TemporaryDirectory() as tmpdir:
                chunk['original_img_path'] = chunk.apply(
                    lambda x: extract(row=x, tmpdir=tmpdir, user_image_mapping_dict=user_image_mapping_dict), axis=1)
                resize_imgs(src_root=f'{tmpdir}/original_pics', dest_root=f'{tmpdir}/resized_pics')
                chunk['img_path'] = chunk['original_img_path'].apply(
                    lambda x: get_resized_path(orig_img_path=x, src_root=f'{tmpdir}/original_pics',
                                               dest_root=f'{tmpdir}/resized_pics'))
                chunk = chunk[['id', 'name', 'screen_name', 'description', 'lang', 'img_path']]
                chunk = chunk.loc[~chunk['img_path'].isnull()].reset_index(drop=True)
                chunk = chunk.dropna()
                df_json = json.loads(chunk.to_json(orient='records'))
                # Run inference
                m3 = M3Inference()  # see docstring for details
                predictions = m3.infer(df_json)  # also see docstring for details
                if predictions:
                    result_file = os.path.join(output_dir, "processed_%s.csv.gz" % str(uuid.uuid4()))

                    rows = []
                    for item in predictions.items():
                        user, pred = item
                        row = {"user": user, "gender": findMax(pred['gender']),
                               "age": findMax(pred['age']), "org": findMax(pred['org'])}
                        rows.append(row)

                    pd.DataFrame(rows).to_csv(result_file, index=False)
    # for user_path in selected_tars_list:
    #     filename = os.path.basename(tar_path.name)
    #     filename_without_ext = os.path.splitext(filename)[0]
    #     sh(f'mkdir -p {resized_dir}/{filename_without_ext}')
    #     tf = tarfile.open(tar_path)
    #     try:
    #         tf = tarfile.open(tar_path)
    #         components_str = tf.getnames[0].count('/')
    #         # untar
    #         sh(f'tar -xf {tar_path.name} -C {resized_dir}/{filename_without_ext} --strip-components {components_str}')
    #         # for each image in {resized_dir}/{filename_without_ext}, resize and remove original file.
    #         resize_imgs(src_root=f'{resized_dir}/{filename_without_ext}',
    #                     dest_root=f'{resized_dir}/{filename_without_ext}_resized')
    #         sh(f'rm -rf {resized_dir}/{filename_without_ext}')
    #         #
    #
    #
    #     except Exception as e:
    #         logger.info(f'Exception "{e}" when processing {tar_path.name}')
    #         continue
