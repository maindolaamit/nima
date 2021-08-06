import argparse
import os

import numpy as np

from nima.config import AVA_DATASET_DIR, WEIGHTS_DIR
from nima.model.data_generator import NimaDataGenerator
from nima.model.model_builder import NIMA
from nima.utils.ava_dataset_utils import load_data, get_rating_columns

input_shape = (224, 224, 3)
x_col, y_cols = 'image_id', get_rating_columns()


def test_model(df, images_dir, model_name, weights_path, input_shape, metrics):
    # Form the NIMA Model
    nima_cnn = NIMA(base_model_name=model_name, weights=weights_path, input_shape=input_shape,
                    metrics=metrics)

    nima_cnn.build()
    # load model from weights
    nima_cnn.model.load_weights(weights_path)
    nima_cnn.compile()

    # Get the generator
    test_generator = NimaDataGenerator(df, images_dir, x_col, y_cols=None,
                                       preprocess_input=nima_cnn.preprocessing_function(),
                                       is_train=False, batch_size=32, )

    steps = int(np.ceil(len(df) / 64))
    predictions = nima_cnn.model.predict(test_generator, steps=steps)

    df['predictions'] = predictions
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model on AVA Dataset')
    parser.add_argument('-d', '--dataset-dir', type=str, default=AVA_DATASET_DIR, required=False,
                        help='Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=False,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-s', '--sample-size', type=int, default=1000, required=False,
                        help='Sample size, None for full size.')
    parser.add_argument('-m', '--metrics', type=list, default=['accuracy'], required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-w', '--weights-path', type=str, default=None, required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=64, required=False, help='Batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=15, required=False,
                        help='Number of epochs, default 10.')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        required=False, help='Verbose, default 0.')
    args = parser.parse_args()

    # Set the AVA Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(
        arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'

    # model to choose, default to mobilenet
    arg_model_name = args.__dict__['model_name']
    arg_weight_path = args.__dict__['weights_path']
    if arg_weight_path is not None:
        assert os.path.isfile(arg_weight_path), 'Invalid weights, does not exists.'

    arg_batch_size = args.__dict__['batch_size']
    arg_sample_size = args.__dict__['sample_size']
    arg_epochs = args.__dict__['epochs']
    arg_verbose = args.__dict__['verbose']
    arg_metrics = args.__dict__['metrics']

    # Load the dataset
    df_train, df_valid, df_test = load_data(arg_dataset_dir, arg_sample_size)
    # Form the NIMA Model
    nima_cnn = NIMA(base_model_name=arg_model_name, weights='imagenet', input_shape=input_shape,
                    metrics=arg_metrics)
    nima_cnn.build()
    # load model weights if existing
    if arg_weight_path is not None:
        nima_cnn.model.load_weights(arg_weight_path)

    nima_cnn.compile()
    nima_cnn.model.summary()

    ava_images_dir = os.path.join(arg_dataset_dir, 'images')
    print(f'Images directory {ava_images_dir}')
    # Get the generator
    train_generator = NimaDataGenerator(df_train, ava_images_dir, x_col, y_cols, img_format='jpg',
                                        preprocess_input=nima_cnn.preprocessing_function(),
                                        is_train=True, batch_size=32, )
    valid_generator = NimaDataGenerator(df_valid, ava_images_dir, x_col, y_cols, img_format='jpg',
                                        preprocess_input=nima_cnn.preprocessing_function(),
                                        is_train=True, batch_size=32, )

    # Train the model
    print("Training Model...")
    result_df, train_weights_file = nima_cnn.train_model(train_generator, valid_generator,
                                                         epochs=arg_epochs, verbose=arg_verbose)

    # print(result_df)

    # Test the model
    print("Testing Model...")
    train_df = test_model(df=df_test, images_dir=ava_images_dir,
                          base_model_name=arg_model_name, weights_path=train_weights_file)
    print(train_df)
