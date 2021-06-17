import os
import re
import shutil

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL_PREFIX = 'http://www.dpchallenge.com/image.php?IMAGE_ID='
DOWNLOAD_DIR = './images/'


def print_msg(message, level=0):
    """ Print the message in formatted way and writes to the log """
    seprator = '\t'
    fmt_msg = f'{level * seprator}{message}'
    print(fmt_msg)


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
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print_msg(f'Image downloaded successfully : {filename}', 2)
    else:
        print_msg('Image could not be retrieved.', 2)


def get_ava_image(url, image_id):
    """
    Download the image from the given url
    :param url:  Image url
    :param image_id: Image id
    """
    filename = os.path.join(DOWNLOAD_DIR, f'{image_id}.jpg')
    # Check if image already exists
    if os.path.isfile(filename):
        print_msg('Image already exists.', 1)
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
            print_msg('Unable to get any ava image in the url.', 1)


if __name__ == '__main__':
    # create directory if not exists
    if not os.path.isdir(DOWNLOAD_DIR):
        os.mkdir(DOWNLOAD_DIR)

    # read the dataframe to fetch image id
    df = pd.read_csv('AVA.txt', sep=' ', header=None)
    # Loop for each image id
    for img_id in df.iloc[:20, 1].tolist():
        img_url = f"{URL_PREFIX}{img_id}"
        print_msg(f'Downloading image ... {img_url}')
        get_ava_image(img_url, img_id)

    print_msg('downloaded AVA dataset.')
