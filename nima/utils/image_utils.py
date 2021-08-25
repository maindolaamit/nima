import os
import warnings
from glob import glob

from PIL import Image, ImageOps, ImageFile
from numpy.random import randint
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from nima.config import INPUT_SHAPE, AVA_DATASET_IMAGES_DIR, PROJECT_ROOT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(image_path, target_size=INPUT_SHAPE, normalize_pixels=False):
    """
    Load the image from disk to Pillow format and then convert to numpy array.
    return numpy array against image after rescaling and normalization.
    :param image_path: image file path
    :param target_size: image rescale size
    :return: Numpy array
    """
    image = load_img(image_path, target_size=(target_size[0], target_size[1]))  # load the image in pillow format
    image = img_to_array(image)  # convert to numpy array
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    if normalize_pixels:
        image = image / 255.0  # Normalize the image pixel value
    return image


def random_crop_image(img, crop_size):
    """
    Randomly crop the image to the given crop size
    :param img: image array
    :param crop_size: crop size (height, width)
    :return: cropped image
    """
    img_h, img_w = img.shape[0], img.shape[1]
    crop_h, crop_w = crop_size[0], crop_size[1]
    assert img_h >= crop_h, f'image height {img_h} should be greater than crop_size {crop_h}'
    assert img_w >= crop_w, f'image width {img_w} should be greater than crop_size {crop_w}'

    x, y = randint(0, img_h - crop_h + 1), randint(0, img_w - crop_w + 1)
    return img[x:x + crop_w, y:y + crop_h, :]


def horizontal_flip(img):
    img = Image.open(img)
    return ImageOps.flip(img)


def filter_valid_images(df, img_directory, x_col, img_format):
    """Keep only dataframe rows with valid filenames
    # Arguments
        df: Pandas dataframe containing filenames in a column
        x_col: string, column in `df` that contains the filenames or filepaths
    # Returns
        absolute paths to image files
    """
    filepaths = df[x_col].map(lambda fname: os.path.join(img_directory, f'{fname}.{img_format}'))
    # mask = filepaths.apply(lambda x: os.path.isfile(x) and is_valid_image(x)) # Commented for performance reasons
    mask = filepaths.apply(lambda x: os.path.isfile(x))
    n_invalid = (~mask).sum()
    if n_invalid:
        warnings.warn(f'Found {n_invalid} invalid image filename(s) in x_col="{x_col}".'
                      f' These filename(s) will be ignored.')
    return df[mask]


def is_valid_image(filename):
    try:
        image = Image.open(filename)
        image.verify()

        image = load_img(filename, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))  # load the image in pillow format
        image = img_to_array(image)  # convert to numpy array
        return True
    except Exception as e:
        return True


def get_images(img_directory):
    files = glob(os.path.join(img_directory, '*.jpg'))
    # files += glob(os.path.join(img_directory, '*.JPG'))
    files += glob(os.path.join(img_directory, '*.bmp'))
    files += glob(os.path.join(img_directory, '*.BMP'))
    return files


def clean_dataset(img_directory):
    files = get_images(img_directory)

    invalid_files = [file for file in files if not is_valid_image(file)]
    print(f"Total number of invalid files : {len(invalid_files)}")
    print(invalid_files[:20])
    for file in invalid_files:
        os.remove(file)


def rename_images(image_dir):
    images = get_images(image_dir)
    if len(images) == 0:
        return

    for i, image in enumerate(images):
        filename = os.path.basename(image)
        new_name = os.path.join(image_dir, f"{i+1}.jpg")
        print(f"renamed '{filename}' to '{new_name}'")
        os.rename(image, new_name)


if __name__ == '__main__':
    # clean_dataset(AVA_DATASET_IMAGES_DIR)
    rename_images(os.path.join(PROJECT_ROOT_DIR, 'test'))
