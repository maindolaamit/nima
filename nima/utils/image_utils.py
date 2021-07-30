from PIL import Image, ImageOps
from numpy.random import randint
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def load_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=(target_size[0], target_size[1]))  # load the image in pillow format
    image = img_to_array(image)  # convert to numpy array
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
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
    assert img_h > crop_h, 'image height should be greater than crop_size'
    assert img_w > crop_w, 'image width should be greater than crop_size'

    x, y = randint(0, img_h - crop_h + 1), randint(0, img_w - crop_w + 1)
    return img[x:x + crop_w, y:y + crop_h, :]


def horizontal_flip(img):
    img = Image.open(img)
    return ImageOps.flip(img)
