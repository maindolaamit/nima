import os

import numpy as np
import tensorflow.keras as keras

from nima.config import INPUT_SHAPE, CROP_SHAPE, print_msg
from nima.utils import image_utils


def normalize_label(label):
    """
    Normalize the given input list of labels
    :return: numpy array
    """
    x = np.array(label)
    return x / x.sum()


class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, df, img_directory, x_col, y_col, preprocess_input, img_format='jpg',
                 num_classes=10, batch_size=32, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE,
                 shuffle=False):
        """
        Takes the dataframe and the path to the image directory and generates the batches of augmented data.
        :param df: Pandas dataframe containing filepaths relative to img_directory
        :param img_directory: path to directory to read images from
        :param batch_size: size of batch of data
        :param x_col: column name in the dataframe to read images from
        :param y_col: column name in the dataframe that has target data
        :param preprocess_input: Base CNN model's preprocessing input function
        :param input_size: dimension that image size get resized to when loaded
        :param crop_size: dimension that image gets randomly cropped to
        :param shuffle: weather to shuffle the data
        """
        self.img_directory = img_directory
        self.x_col, self.y_col = x_col, y_col
        self.batch_size = batch_size
        self.crop_size, self.input_size = crop_size, input_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.model_preprocess_input = preprocess_input
        self.img_format = img_format

        # Filter dataframe for valid images only
        self.df = image_utils.filter_valid_images(df.copy(), img_directory, x_col, img_format)
        print_msg(f'Found {len(self.df)} valid image filenames belonging to {self.num_classes} classes.', 1)

        self.on_epoch_end()  # set the indexes and shuffle at the start if true

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        """
        The index passed into the function will be done by the fit function while training.
        Note : Take batch samples length not the batch size for initialization otherwise in the last epoch
        number of rows returned will be more than length of self.indexes and predict will fail in
        """
        # Extract samples from df based on the index passed by fit method
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Reset index to not exceed the batch size
        batch_samples = self.df.iloc[batch_indexes].copy().reset_index(drop=True)
        # Initialization for faster processing
        x = np.empty((len(batch_samples), *self.crop_size))
        y = np.empty((len(batch_samples), self.num_classes))
        # loop for the images in the sample and modify
        for i, row in batch_samples.iterrows():
            # Load the image and resize
            img_path = os.path.join(self.img_directory, f'{row[self.x_col]}.{self.img_format}')
            img = image_utils.load_image(img_path, self.input_size)
            if img is not None:
                # Modify image only for training purpose
                img = image_utils.random_crop_image(img, self.crop_size)  # crop the image
                # randomly flip image horizontal
                if np.random.random() > 0.5:
                    img = np.fliplr(img)
                x[i] = img
            y[i] = normalize_label(row[self.y_col])
        # apply base network's preprocessing on the 4D numpy array
        x = self.model_preprocess_input(x)
        # return the image and labels
        return x, y

    def on_epoch_end(self):
        """Will be called before and after each epoch"""
        self.indexes = np.arange(len(self.df))  # Create an index
        if self.shuffle:
            # Updates indexes after each epoch
            np.random.shuffle(self.indexes)


class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, df, img_directory, x_col, y_col, preprocess_input, img_format='jpg', num_classes=10,
                 batch_size=32, input_size=INPUT_SHAPE):
        """
        Takes the dataframe and the path to the image directory and generates the batches of augmented data.
        :param df: Pandas dataframe containing filepaths relative to img_directory
        :param img_directory: path to directory to read images from
        :param batch_size: size of batch of data
        :param x_col: column name in the dataframe to read images from
        :param y_col: column name in the dataframe that has target data
        :param preprocess_input: Base CNN model's preprocessing input function
        :param input_size: dimension that image size get resized to when loaded
        """
        self.input_size = input_size
        self.img_directory = img_directory
        self.x_col, self.y_col = x_col, y_col
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model_preprocess_input = preprocess_input
        self.img_format = img_format

        # Filter dataframe for valid images only
        self.df = image_utils.filter_valid_images(df.copy(), img_directory, x_col, img_format)
        print_msg(f'Found {len(self.df)} valid image filenames belonging to {self.num_classes} classes.', 1)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        """
        The index passed into the function will be done by the fit function while training.
        Note : Take batch samples length not the batch size for initialization otherwise in the last epoch
        number of rows returned will be more than length of self.indexes and predict will fail in
        """
        # Extract samples from df based on the index passed by fit method
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Reset index to not exceed the batch size
        batch_samples = self.df.iloc[batch_indexes].copy().reset_index(drop=True)
        # Initialization for faster processing
        x = np.empty((len(batch_samples), *self.input_size))
        y = np.empty((len(batch_samples), self.num_classes))
        # loop for the images in the sample and modify
        for i, row in batch_samples.iterrows():
            # Load the image and resize
            img_path = os.path.join(self.img_directory, f'{row[self.x_col]}.{self.img_format}')
            img = image_utils.load_image(img_path, self.input_size)
            if img is not None:
                x[i] = img
            if self.y_col is not None:
                y[i] = normalize_label(row[self.y_col])
        # apply base network's preprocessing on the 4D numpy array
        x = self.model_preprocess_input(x)
        # return the image and labels
        return x, y

    def on_epoch_end(self):
        """Will be called before and after each epoch"""
        self.indexes = np.arange(len(self.df))  # Create an index
