import os

import pandas as pd
from sklearn.model_selection import train_test_split

from nima.config import TID_DATASET_DIR, print_msg

columns = [
    "rating",
    "image_id",
]


def get_mos_df(dataset_dir=None):
    """
    Get the DataFrame of mos_with_names.txt file, having Opinion score of the images.
    :param dataset_dir: Image dataset directory
    :return: Pandas DataFrame
    """
    if dataset_dir is None:
        dataset_dir = TID_DATASET_DIR
    df = pd.read_csv(os.path.join(dataset_dir, 'mos_with_names.txt'), sep=' ', header=None, names=columns)
    df['rating'] = df['rating']
    df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
    return df


def load_tid_data(tid_dataset=None, sample_size=None):
    """
    Get the Training, Validation and Testing data as Dataframes.
    :param tid_dataset: TID Dataset directory.
    :param sample_size: Number of samples to use
    :return:
    """
    if tid_dataset is None:
        tid_dataset = TID_DATASET_DIR

    tid_df = get_mos_df(tid_dataset)
    if sample_size is None or sample_size > len(tid_df):
        sample_size = len(tid_df)
    print_msg(f'Number of samples picked {sample_size}', 1)
    df = tid_df.sample(n=sample_size).reset_index(drop=True)

    df_train, df_test = train_test_split(df, test_size=0.10, shuffle=True, random_state=1024)
    df_train, df_valid = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=1024)

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)
