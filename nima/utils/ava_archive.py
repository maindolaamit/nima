import argparse
import os
from pathlib import Path
from zipfile import ZipFile

import modin.pandas as pd
import numpy as np

from nima.utils.ava_preprocess import columns

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AVA_DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')
IMAGE_DIR = os.path.join(AVA_DATASET_DIR, 'images')


def get_ava_df(ava_file):
    ava_df = pd.read_csv(ava_file, sep=' ', header=None, names=columns, )
    ava_df['batch_id'] = np.ceil(ava_df['image_id'] / 10000)
    ava_df['batch_id'] = ava_df['batch_id'].astype(int)
    return ava_df[['image_id', 'batch_id']]


def archive_images(image_df, batch_id):
    archive_dir = os.path.join(AVA_DATASET_DIR, 'archive')
    if not os.path.isdir(archive_dir):
        os.mkdir(archive_dir)
    cwd = os.getcwd()
    os.chdir(archive_dir)

    # zip all the files
    zipObj = ZipFile.open(f'{batch_id}.zip', 'w')
    image_df.apply(lambda x: zipObj.write(os.path.join(IMAGE_DIR, x + ".jpg")))
    zipObj.close()

    # delete the files
    # image_df.apply(lambda x: os.remove(os.path.join(IMAGE_DIR, x + ".jpg")))
    os.chdir(cwd)


def __main__(self):
    parser = argparse.ArgumentParser(description='Archive AVA dataset Images')
    parser.add_argument('-d', '--dataset-dir', help='Image download directory.', required=False)
    args = parser.parse_args()

    arg_dataset_dir = args.__dict__['download_dir']
    print(f'Download directory  : {arg_dataset_dir}')
    global AVA_DATASET_DIR
    if arg_dataset_dir is not None:
        AVA_DATASET_DIR = arg_dataset_dir
    else:
        AVA_DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')

    ava_file = os.path.join(AVA_DATASET_DIR, 'AVA.txt')
    df = get_ava_df(ava_file)

    # archive the files in batch
    for batch_id in df['batch_id'].unique().tolist():
        archive_images(df[df['batch_id'] == batch_id], batch_id)
