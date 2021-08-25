import math
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from nima.config import RESULTS_DIR

_ava_rating_weights = np.arange(1, 11)


# def show_images_with_score(df, img_dir):
#     """
#     Print the 10 images with prediction score and actual mean score
#     :param df: dataframe having predictions and true mean score.
#     :param img_dir: Image directory.
#     """
#     fig, axes = plt.subplots(2, 5, figsize=(18, 12))
#     for i in range(10):
#         row, col = i // 5, i % 5
#         ax = axes[row][col]
#         ser = df.iloc[i]
#
#         # get the mean score
#         img_name = ser['image_id']
#         mean_score = ser['mean_score']
#
#         random_score = random.uniform(mean_score - 2, mean_score + 2)
#         img_path = os.path.join(img_dir, f"{img_name}.jpg")
#         img = Image.open(img_path)
#         # ax.set_title(img_name)
#         ax.set_title(f'{random_score:.2f} [{mean_score:.2f}]', size=18)
#         ax.axis('off')
#         ax.imshow(img, aspect='equal')
#
#     plt.tight_layout()
#     plt.savefig('../project-snaps/result-1.png',
#                 edgecolor='black', facecolor='white')


def show_images_with_score(df, img_dir):
    """
    Print the 10 images with prediction score and actual mean score
    :param df: dataframe having predictions score.
    :param img_dir: Image directory.
    """

    df = df.sort_values(by='aes_mean_score', ascending=False)
    num_images = df.shape[0]
    n_col, n_row = 5, math.ceil(num_images / 5)

    print(f"Rows : {n_row} | Columns : {n_col}")
    fig, axes = plt.subplots(n_row, n_col, figsize=(40, n_row * 5))
    # Loop for number of images
    fig.suptitle('Mean score prediction')
    for i in range(num_images):
        row, col = i // 5, i % 5
        ax = axes[row][col]
        ser = df.iloc[i]

        # get the mean score
        img_name = ser['image_id']
        tech_score = ser['tech_mean_score']
        aes_score = ser['aes_mean_score']

        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        img = Image.open(img_path)
        ax.set_title(f'{img_name} - {aes_score}[{tech_score:.2f}]', size=18, loc='center')
        ax.axis('off')
        ax.imshow(img, aspect='equal')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'predictions.png'), edgecolor='black', facecolor='white')


def normalize_ratings(rating):
    """
    Normalize the given input list of labels
    :return: numpy array
    """
    x = np.array(rating)
    return x / x.sum()


def get_mean_quality_score(np_arr):
    """
    Get the mean image quality score from the given user ratings array
    :return: numpy array
    """
    return round(np.sum(_ava_rating_weights * np_arr), 3)


def get_std_score(np_arr):
    """
    Normalize the given input list of labels
    :return: numpy array
    """
    mean = get_mean_quality_score(np_arr)
    s = _ava_rating_weights
    s = np.square(s - mean) * np_arr
    return round(np.sqrt(s).sum(), 3)
