"""AVA Dataset Downloader

Usage:
  ava_downloader.py (-d | --dir) <full_path>
  ava_downloader.py (-h | --help)
  ava_downloader.py --version

Options:
  -h --help              Show this screen.
  --version              Show version.
  -d --download-dir      Download directory of the project, default is the current script directory ../../data/AVA/images

"""
import argparse
import os
import re
import shutil

import pandas as pd
import requests
from bs4 import BeautifulSoup

from nima.config import AVA_DATASET_DIR, print_msg

URL_PREFIX = 'http://www.dpchallenge.com/image.php?IMAGE_ID='
DATASET_DIR = AVA_DATASET_DIR
DOWNLOAD_DIR = os.path.join(AVA_DATASET_DIR, 'images')
img_count = 0


def get_request_soup(url):
    """ This method will fetch the request content from passed url and will return the soup element"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/76.0.3809.132 Safari/537.36"}
    url_hdr = headers

    # Check if use fake agent is to be used, saves time
    page_content = requests.get(url=url, headers=url_hdr).content
    return BeautifulSoup(page_content, 'html.parser')


def download_image(image_url, filename):
    """
    Download the given image url to local copy filename
    :param image_url:  Image url
    :param filename:  Local Image url
    """
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream=True)
    # Check if the image was retrieved successfully
    global img_count
    if r.status_code == 200:
        img_count += 1
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print_msg(f'Image downloaded successfully : {filename}', 1)
    else:
        print_msg(f'Image could not be retrieved : {image_url}.', 1)


def get_ava_image(url, image_id):
    """
    Download the image from the given url
    :param url:  Image url
    :param image_id: Image id
    """
    filename = os.path.join(DOWNLOAD_DIR, f'{image_id}.jpg')
    # Check if image already exists
    if os.path.isfile(filename):
        # print_msg('Image already exists.', 1)
        return
    else:
        soup = get_request_soup(url)
        soup.prettify()
        # Find the images in the given page and filter the one matches id
        reg = f'.*{image_id}.jpg'
        images_source = [f"http:{x.get('src')}" for x in soup.findAll('img')]
        images_result = list(filter(lambda x: re.match(reg, x), images_source))
        # Download the image, first image from List
        if images_result is not None and len(images_result) > 0:
            download_image(images_result[0], filename)
        else:
            print_msg(f'Unable to get any ava image for url : {url}', 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download AVA dataset Images')
    parser.add_argument('-d', '--download-dir', help='Image download directory.', required=False)
    args = parser.parse_args()

    arg_download_dir = args.__dict__['download_dir']
    if arg_download_dir is not None:
        DATASET_DIR = arg_download_dir
    else:
        DATASET_DIR = AVA_DATASET_DIR

    DOWNLOAD_DIR = os.path.join(DATASET_DIR, 'images')
    ava_file = os.path.join(DATASET_DIR, 'AVA.txt')

    # create directory if not exists
    if not os.path.isdir(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)

    print(f'Download directory  : {DOWNLOAD_DIR}')
    # read the dataframe to fetch image id
    df = pd.read_csv(ava_file, sep=' ', header=None)
    # Loop for each image id
    print_msg('Downloading dataset')
    for img_id in df.iloc[:, 1].tolist():
        img_url = f"{URL_PREFIX}{img_id}"
        # print_msg(f'Downloading image ... {img_url}')
        get_ava_image(img_url, img_id)
        # todo : temp fix
        # break the loop to avoid blocking from image server
        if img_count >= 100:
            break

    print_msg(f'downloaded AVA dataset, image count {img_count}.')
