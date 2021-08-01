import argparse
import os
from nima.config import AVA_DATASET_DIR
from zipfile import ZipFile

import pandas as pd
import numpy as np

columns = [
    "index",
    "image_id",
    "count_rating_1",
    "count_rating_2",
    "count_rating_3",
    "count_rating_4",
    "count_rating_5",
    "count_rating_6",
    "count_rating_7",
    "count_rating_8",
    "count_rating_9",
    "count_rating_10",
    "tag_1",
    "tag_2",
    "challange_id",
]

# PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = AVA_DATASET_DIR


def get_ava_df(ava_file):
    ava_df = pd.read_csv(ava_file, sep=' ', header=None, names=columns, )
    ava_df['batch_id'] = np.ceil(ava_df['image_id'] / 10000)
    ava_df['batch_id'] = ava_df['batch_id'].astype(int)
    return ava_df[['image_id', 'batch_id']]


def archive_images(image_df, batch_id):
    archive_dir = os.path.join(DATASET_DIR, 'archive')
    image_dir = os.path.join(DATASET_DIR, 'images')
    if not os.path.isdir(archive_dir):
        os.mkdir(archive_dir)
    cwd = os.getcwd()
    os.chdir(archive_dir)

    file_count = 0
    # zip all the files in the batch
    zip_obj_name = f'{batch_id}.zip'
    if os.path.isfile(zip_obj_name):
        os.remove(zip_obj_name)

    zip_obj = ZipFile(zip_obj_name, 'w')
    for i, row in image_df.iterrows():
        image_path = os.path.join(image_dir, f"{row['image_id']}.jpg")
        if os.path.isfile(image_path):
            zip_obj.write(image_path, os.path.basename(image_path))
            file_count += 1
    zip_obj.close()

    # Delete if no image written
    if file_count > 0:
        print(f"\t{zip_obj_name} : Total files written - {file_count}")
    else:
        os.remove(zip_obj_name)

    os.chdir(cwd)
    return os.path.join(archive_dir, zip_obj_name)


"""
Usage :  python nima/utils/ava_archive.py -d E:\AVA_dataset\
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Archive AVA dataset Images')
    parser.add_argument('-d', '--dataset-dir', help='Images directory.', required=False)
    args = parser.parse_args()

    arg_dataset_dir = args.__dict__['dataset_dir']
    if arg_dataset_dir is not None:
        DATASET_DIR = arg_dataset_dir

    print(f'Dataset directory  : {DATASET_DIR}')
    ava_file = os.path.join(DATASET_DIR, 'AVA.txt')
    df = get_ava_df(ava_file)

    # archive the files in batch
    for batch_id in df['batch_id'].unique().tolist():
        zip_file = archive_images(df[df['batch_id'] == batch_id], batch_id)
