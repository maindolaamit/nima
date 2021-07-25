import os
import warnings

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from nima.utils import image_utils


class NimaDataGenerator(keras.utils.Sequence):
    def __init__(self, df, img_directory, x_col, y_col, preprocess_input,
                 is_train=False, batch_size=32, input_size=(224, 224, 3), crop_size=(224, 224),
                 shuffle=True):
        """
        Takes the dataframe and the path to the image directory and generates the batches of augmented data.
        :param df: Pandas dataframe containing filepaths relative to img_directory
        :param img_directory: path to directory to read images from
        :param batch_size: size of batch of data
        :param x_col: column name in the dataframe to read images from
        :param y_col: column name in the dataframe that has target data
        :param is_train: Identifier to detect if the data generator is for train or test
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
        self.num_classes = 10
        self.preprocess_input = preprocess_input
        self.is_train = is_train
        # Filter dataframe for valid images only
        self.df = self._filter_valid_filepaths(df.copy(), x_col)
        self.indexes = np.arange(len(self.df))  # Create an index
        print(f'Found {len(self.df)} valid image filenames belonging to {self.num_classes} classes.')

    def __len__(self):
        return np.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        """The index passed into the function will be done by the fit function while training."""
        # Extract samples from df based on the index passed by fit method
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = self.df.iloc[batch_indexes]
        # Initialization
        X = np.empty((self.batch_size, *self.crop_size, 3))
        y = np.empty((self.batch_size, *self.num_classes))
        # loop for the images in the sample and modify
        for i, row in batch_samples.iterrows():
            # Load the image and resize
            img_path = os.path.join(self.img_directory, row[self.x_col] + '.jpg')
            img = image_utils.load_image(img_path, self.input_size)
            # Modify image only for training purpose
            if self.is_train:
                # horizontal flip and crop randomly
                img = image_utils.random_crop_image(img, self.crop_size)
                if (np.random.random() > 0.5) and self.flip_horizontal:
                    img = np.fliplr(img)
            X[i] = img
            y[i] = self._normalize_label(row[self.y_col])
        # apply base network's preprocessing on the 4D numpy array
        # X = self.preprocess_input(X)
        # return the image and labels
        return X, y

    def _normalize_label(label):
        """
        Normalize the given input list of labels
        :return: numpy array
        """
        x = np.array(label)
        return x / x.sum()

    def on_epoch_end(self):
        """Will be called before and after each epoch"""
        if self.shuffle:
            # Updates indexes after each epoch
            np.random.shuffle(self.indexes)

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames
        # Arguments
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths
        # Returns
            absolute paths to image files
        """
        filepaths = df[x_col].map(lambda fname: os.path.join(self.img_directory, f'{fname}.jpg'))
        mask = filepaths.apply(lambda x: os.path.isfile(x))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(f'Found {n_invalid} invalid image filename(s) in x_col="{x_col}".'
                          f' These filename(s) will be ignored.')
        return df[mask]

