import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nima.config import TID_DATASET_DIR, print_msg

columns = [
    "rating",
    "image_id",
]

TID_MAX_MEAN_SCORE = 9


def _get_dataset_dir(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = TID_DATASET_DIR
    return dataset_dir


def make_mos_csv(dataset_dir=None):
    dataset_dir = _get_dataset_dir(dataset_dir)
    df_std = pd.read_csv(os.path.join(dataset_dir, 'mos_std.txt'), header=None, names=['std'])
    df = pd.read_csv(os.path.join(dataset_dir, 'mos_with_names.txt'), sep=' ', header=None,
                     names=['mean', 'image_id'])
    df_merge = pd.concat([df, df_std], axis=1)
    df_merge[['image_id', 'mean', 'std']].to_csv(os.path.join(dataset_dir, 'mos.csv'), index=False)


def get_mos_df(dataset_dir=None):
    """
    Get the DataFrame of mos_with_names.txt file, having Opinion score of the images.
    :param dataset_dir: Image dataset directory
    :return: Pandas DataFrame
    """
    dataset_dir = _get_dataset_dir(dataset_dir)
    df = pd.read_csv(os.path.join(dataset_dir, 'mos_with_names.txt'), sep=' ', header=None, names=columns)
    # df['rating'] = df['rating']
    df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
    return df


def get_mos_csv_df(dataset_dir=None):
    """
    Reads the mos.csv file and forms the Dataframe.
    :param dataset_dir: TID2013 dataset directory.
    :return: Pandas DataFrame
    """
    dataset_dir = _get_dataset_dir(dataset_dir)
    df = pd.read_csv(os.path.join(dataset_dir, 'mos.csv'))
    df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
    return df


def load_tid_data(dataset_dir=None, sample_size=None, require_validation_data=True, ):
    """
    Get the Training, Validation and Testing data as Dataframes.
    :param require_validation_data: If yes validation data will also be given
    :param dataset_dir: TID Dataset directory.
    :param sample_size: Number of samples to use
    :return:
    """
    dataset_dir = _get_dataset_dir(dataset_dir)

    tid_df = get_mos_csv_df(dataset_dir)

    num_brackets = 3
    if sample_size is None or sample_size > len(tid_df):
        sample_size = len(tid_df)
        num_brackets = 1

    # Add label for stratification
    tid_df['_label'] = np.floor(tid_df['mean'] / num_brackets)
    print_msg(f'Number of samples picked {sample_size}', 1)
    tid_df = tid_df.sample(n=sample_size).reset_index(drop=True)
    df_train, df_test = train_test_split(tid_df, test_size=0.10, shuffle=True, random_state=1024,
                                         stratify=tid_df['_label'])
    df_train = df_train.drop(labels='_label', axis=1).reset_index(drop=True)
    df_test = df_test.drop('_label', axis=1).reset_index(drop=True)

    # if require_validation_data:
    df_train, df_valid = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=1024)

    df_train['mean'] = df_train['mean'] / TID_MAX_MEAN_SCORE  # Normalize mean
    df_valid['mean'] = df_valid['mean'] / TID_MAX_MEAN_SCORE  # Normalize mean
    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test
