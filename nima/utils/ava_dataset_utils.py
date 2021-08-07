import os
from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split

from nima.utils.ava_downloader import AVA_DATASET_DIR

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
    """
    Get the list of Ratings column
    :return: list of ratings column
    """
    global __rating_columns
    if __rating_columns is None:
        __rating_columns = [x for x in columns if x.startswith('count_rating')]
    return __rating_columns


def _get_present_image_namess_df(image_dir):
    """
    Fetch all the images in the given directory and return in Pandas DataFrame
    :rtype: Pandas dataframe having names of all the images present in the path
    """
    image_files = [os.path.basename(name) for name in glob(os.path.join(image_dir, '*.jpg'))]
    df_image = pd.DataFrame(image_files, columns=['image_id'])
    return df_image


def make_ava_csv(dataset_dir=None):
    """
    Look for the images present in the given dataset directory and create/update the AVA.csv file
    :param dataset_dir: Path of the AVA Dataset, should have a folder images having actual images in it.
    """
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR

    # Fetch the list of images and Merge the two dataframe on id
    print('Getting present images list')
    images_dir = os.path.join(dataset_dir, 'images')
    df_images = _get_present_image_namess_df(images_dir)
    df_orig = get_original_ava_df(dataset_dir)
    print('creating dataframe of images name')
    images_present_list = df_images['image_id'].apply(lambda x: x.split('.')[0]).astype(int).to_list()
    df_images_present = df_orig[df_orig['image_id'].isin(images_present_list)]
    # Save the dataframe to csv
    df_images_present.to_csv(os.path.join(dataset_dir, 'AVA.csv'), sep=',', header=True, index=False)


def get_ava_csv_df(dataset_dir=None):
    """
    Returns the Pandas DataFrame from AVA.csv file
    :param dataset_dir: AVA Dataset directory.
    :return: Pandas DataFrame from AVA.csv
    """
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR
    df = pd.read_csv(os.path.join(dataset_dir, 'AVA.csv'))
    return df


def get_original_ava_df(dataset_dir=None):
    """
    Returns the Pandas DataFrame from original file AVA.txt
    :param dataset_dir: AVA dataset directory
    :return: Pandas DataFrame
    """
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR

    ava_file = os.path.join(dataset_dir, 'AVA.txt')
    return pd.read_csv(ava_file, sep=' ', header=None, names=columns, )


def __get_max_rating(df_row):
    """
    Get the max rating from the passed DataFrame row of original ava.txt
    :param df_row: DataFrame Row
    :return: Pandas Series having max rating value
    """
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


def get_tags_df(dataset_dir=None):
    """
    Get the DataFrame of tags.txt file, having tags information of the images.
    :param dataset_dir: AVA Dataset directory.
    :return: Pandas DataFrame
    """
    if dataset_dir is None:
        dataset_dir = AVA_DATASET_DIR
    df = pd.read_csv(os.path.join(dataset_dir, 'tags.txt'), sep=' ', header=None, names=['id', 'label'])
    df['id'] = df['id'].astype(int)
    df.sort_values(by=['id'], inplace=True)
    return df


def load_data(ava_dataset_dir=None, sample_size=None):
    """
    Returns the pandas DataFrame for Training Data, Test Data and Validation Data.
    :param ava_dataset_dir: AVA Dataset directory.
    :param sample_size: No. of Samples to pick from DataFrame.
    :return: Train DataFrame, Validation DataFrame, Test DataFrame
    """
    if ava_dataset_dir is None:
        ava_dataset_dir = AVA_DATASET_DIR

    count_columns = get_rating_columns()  # get the columns representing ratings
    ava_csv_df = get_ava_csv_df(ava_dataset_dir)  # Get the AVA csv dataframe

    if sample_size is None:
        sample_size = len(ava_csv_df)

    keep_columns = ['image_id'] + count_columns
    if sample_size is None or sample_size > len(ava_csv_df):
        sample_size = len(ava_csv_df)
    print(f'Number of samples picked {sample_size}')
    df = ava_csv_df[keep_columns].sample(n=sample_size).reset_index(drop=True)

    df_train, df_test = train_test_split(df, test_size=0.05, shuffle=True, random_state=1024)
    df_train, df_valid = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=1024)

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)


if __name__ == '__main__':
    make_ava_csv('E:\AVA_dataset')
