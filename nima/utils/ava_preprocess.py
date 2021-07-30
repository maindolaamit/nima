import pandas as pd
import os
from pathlib import Path
from glob import glob

from nima.utils.ava_downloader import AVA_DATASET_DIR

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AVA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')
AVA_DATA_DIR = os.path.join(AVA_DIR, 'images')
AVA_FILE = os.path.join(AVA_DIR, 'AVA.txt')
AVA_CSV = os.path.join(AVA_DIR, 'AVA.csv')

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

# Save the ratings
__rating_columns = None


def get_rating_columns():
    global __rating_columns
    if __rating_columns is None:
        __rating_columns = [x for x in columns if x.startswith('count_rating')]
    return __rating_columns


def get_present_image_names_df(image_dir):
    image_files = [os.path.basename(name) for name in glob(os.path.join(image_dir, '*.jpg'))]
    df_image = pd.DataFrame(image_files, columns=['image_id'])
    return df_image


def make_ava_csv(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR

    # Fetch the list of images and Merge the two dataframe on id
    print('Getting present images list')
    images_dir = os.path.join(dataset_dir, 'images')
    df_images = get_present_image_names_df(images_dir)
    df_orig = get_original_ava_df(dataset_dir)
    print('creating dataframe of images name')
    images_present_list = df_images['image_id'].apply(lambda x: x.split('.')[0]).astype(int).to_list()
    df_images_present = df_orig[df_orig['image_id'].isin(images_present_list)]
    # Save the dataframe to csv
    df_images_present.to_csv(os.path.join(dataset_dir, 'AVA.csv'), sep=',', header=True, index=False)


def get_ava_csv_df(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR
    df = pd.read_csv(os.path.join(dataset_dir, 'AVA.csv'))
    return df


def get_original_ava_df(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR

    ava_file = os.path.join(dataset_dir, 'AVA.txt')
    return pd.read_csv(AVA_FILE, sep=' ', header=None, names=columns, )


def __get_max_rating(df_row):
    row = df_row[get_rating_columns()]
    max_value_id = row.idxmax()
    max_rating = max_value_id.replace('count_rating_', '')
    return max_rating


def get_orig_df_with_max_rating(dataset_path=None):
    df = get_original_ava_df()
    df['rating'] = df.apply(lambda row: __get_max_rating(row), axis=1)
    return df


def get_csv_df_with_max_rating(dataset_dir=None):
    df = get_ava_csv_df(dataset_dir)
    df['rating'] = df.apply(lambda row: __get_max_rating(row), axis=1)
    return df


def get_tags_df():
    df = pd.read_csv(os.path.join(AVA_DIR, 'tags.txt'), sep=' ', header=None, names=['id', 'label'])
    df['id'] = df['id'].astype(int)
    df.sort_values(by=['id'], inplace=True)
    return df


if __name__ == '__main__':
    make_ava_csv('E:\AVA_dataset')
