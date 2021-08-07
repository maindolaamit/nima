import argparse
import os

import numpy as np

from nima.config import INPUT_SHAPE, DATASET_DIR, CROP_SHAPE
from nima.model.data_generator import TrainDataGenerator, TestDataGenerator
from nima.model.model_builder import NIMA
from nima.utils.ava_downloader import print_msg


def train_aesthetic_model(p_model_name, p_dataset_dir, p_sample_size, p_weight_path,
                          p_batch_size, p_metrics, p_epochs, p_verbose):
    from nima.utils.ava_dataset_utils import load_data, get_rating_columns
    ava_dataset_dir = os.path.join(p_dataset_dir, 'AVA')
    ava_images_dir = os.path.join(ava_dataset_dir, 'images')
    img_format = 'jpg'
    print_msg(f'Images directory {ava_images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', get_rating_columns()
    df_train, df_valid, df_test = load_data(p_dataset_dir, p_sample_size)
    assert len(df_train) > 0 and len(df_valid) > 0 and len(df_test) > 0, 'Empty dataframe'
    train_batch_size = valid_batch_size = p_batch_size
    test_batch_size = min(p_batch_size, 32, len(df_test))

    # Form the NIMA Aesthetic Model
    nima_aesthetic_cnn = NIMA(base_model_name=p_model_name, weights='imagenet', model_type='aesthetic',
                              input_shape=INPUT_SHAPE, metrics=p_metrics)

    # Build the model for training
    nima_aesthetic_cnn.build()
    # load model weights if existing
    if p_weight_path is not None:
        print_msg(f'Using weight {p_weight_path}')
        nima_aesthetic_cnn.model.load_weights(p_weight_path)

    nima_aesthetic_cnn.compile()
    nima_aesthetic_cnn.model.summary()

    # Get the generator
    train_generator = TrainDataGenerator(df_train, ava_images_dir, x_col=x_col, y_col=y_cols,
                                         img_format=img_format, num_classes=10,
                                         preprocess_input=nima_aesthetic_cnn.preprocessing_function(),
                                         batch_size=train_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)
    valid_generator = TrainDataGenerator(df_valid, ava_images_dir, x_col=x_col, y_col=y_cols,
                                         img_format=img_format, num_classes=10,
                                         preprocess_input=nima_aesthetic_cnn.preprocessing_function(),
                                         batch_size=train_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)

    # Train the model
    print_msg("Training Aesthetic Model...")
    train_result_df, train_weights_file = nima_aesthetic_cnn.train_model(train_generator, valid_generator,
                                                                         epochs=p_epochs, verbose=p_verbose)

    # Test the model
    print_msg("Testing Model...")
    # Get the generator
    test_generator = TestDataGenerator(df_test, ava_images_dir, x_col=x_col, y_col=None,
                                       img_format=img_format, num_classes=10,
                                       preprocess_input=nima_aesthetic_cnn.preprocessing_function(),
                                       input_size=INPUT_SHAPE, batch_size=test_batch_size)

    predictions = nima_aesthetic_cnn.model.predict(test_generator)
    print_msg(predictions.shape)

    df_test['predictions'] = predictions
    return train_result_df, df_test, train_weights_file


def train_technical_model(p_model_name, p_dataset_dir, p_sample_size, p_weight_path,
                          p_batch_size, p_metrics, p_epochs, p_verbose):
    from nima.utils.tid_dataset_utils import load_tid_data
    tid_dataset_dir = os.path.join(p_dataset_dir, 'tid2013')
    tid_images_dir = os.path.join(tid_dataset_dir, 'distorted_images')
    img_format = 'bmp'
    print_msg(f'Images directory {tid_images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', 'rating'
    df_train, df_valid, df_test = load_tid_data(tid_dataset_dir, p_sample_size)
    assert len(df_train) > 0 and len(df_valid) > 0 and len(df_test) > 0, 'Empty dataframe'
    print_msg(df_train.iloc[0])
    train_batch_size = valid_batch_size = p_batch_size
    test_batch_size = min(p_batch_size, 32, len(df_test))

    # Form the NIMA Aesthetic Model
    nima_technical_cnn = NIMA(base_model_name=p_model_name, weights='imagenet', model_type='technical',
                              input_shape=INPUT_SHAPE, metrics=p_metrics)

    # Build the model for training
    nima_technical_cnn.build()
    # load model weights if existing
    if p_weight_path is not None:
        print_msg(f'Using weight {p_weight_path}', 1)
        nima_technical_cnn.model.load_weights(p_weight_path)

    nima_technical_cnn.compile(train_layers=False)
    nima_technical_cnn.model.summary()

    # Get the generator
    train_generator = TrainDataGenerator(df_train, tid_images_dir, x_col=x_col, y_col=y_cols,
                                         img_format=img_format, num_classes=1,
                                         preprocess_input=nima_technical_cnn.preprocessing_function(),
                                         batch_size=train_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)
    valid_generator = TrainDataGenerator(df_valid, tid_images_dir, x_col, y_cols, img_format=img_format, num_classes=1,
                                         preprocess_input=nima_technical_cnn.preprocessing_function(),
                                         batch_size=train_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)

    # Train the model
    print_msg("Training Technical Model...")
    print_msg(f'Training Batch size {train_batch_size}', 1)
    train_result_df, train_weights_file = nima_technical_cnn.train_model(train_generator, valid_generator,
                                                                         epochs=p_epochs, verbose=p_verbose)

    # Test the model
    print_msg("Testing Model...")
    print_msg(f'Testing Batch size {test_batch_size}', 1)
    # Get the generator
    test_generator = TestDataGenerator(df_test, tid_images_dir, x_col=x_col, y_col=None,
                                       img_format=img_format, num_classes=1,
                                       preprocess_input=nima_technical_cnn.preprocessing_function(),
                                       batch_size=test_batch_size, input_size=INPUT_SHAPE)

    test_steps = np.ceil(len(test_generator) / test_batch_size)
    predictions = nima_technical_cnn.model.predict(test_generator, steps=test_steps)
    print_msg(predictions)
    df_test['predictions'] = predictions
    return train_result_df, df_test, train_weights_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model on AVA Dataset')
    parser.add_argument('-d', '--dataset-dir', type=str, default=DATASET_DIR, required=False,
                        help='Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=False,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-s', '--sample-size', type=int, default=None, required=False,
                        help='Sample size, None for full size.')
    parser.add_argument('-m', '--metrics', type=list, default=['accuracy'], required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-t', '--model-type', type=str, default='aesthetic', required=False,
                        help='Model type to train aesthetic/technical/both.')
    parser.add_argument('-wa', '--aes-weights-path', type=str, default=None, required=False,
                        help='Aesthetic Weights file path, if any.')
    parser.add_argument('-wt', '--tech-weights-path', type=str, default=None, required=False,
                        help='Technical Weights file path, if any.')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=64, required=False, help='Batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=15, required=False,
                        help='Number of epochs, default 10.')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        required=False, help='Verbose, default 0.')
    args = parser.parse_args()

    # Set the Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'

    # model to choose, default to mobilenet
    arg_model_name = args.__dict__['model_name']
    arg_model_type = args.__dict__['model_type']
    arg_aes_weight_path = args.__dict__['aes_weights_path']
    if arg_aes_weight_path is not None:
        assert os.path.isfile(arg_aes_weight_path), 'Invalid Aesthetic weights, does not exists.'

    arg_tech_weight_path = args.__dict__['aes_weights_path']
    if arg_tech_weight_path is not None:
        assert os.path.isfile(arg_tech_weight_path), 'Invalid Technical weights, does not exists.'

    arg_batch_size = args.__dict__['batch_size']
    arg_sample_size = args.__dict__['sample_size']
    arg_epochs = args.__dict__['epochs']
    arg_verbose = args.__dict__['verbose']
    arg_metrics = args.__dict__['metrics']

    # Train the aesthetic model
    if arg_model_type == ['aesthetic', 'both']:
        aes_train_result_df, aes_test_df, aes_weight_file = train_aesthetic_model(p_model_name=arg_model_name,
                                                                                  p_dataset_dir=arg_dataset_dir,
                                                                                  p_sample_size=arg_sample_size,
                                                                                  p_weight_path=arg_aes_weight_path,
                                                                                  p_batch_size=arg_batch_size,
                                                                                  p_metrics=arg_metrics,
                                                                                  p_epochs=arg_epochs,
                                                                                  p_verbose=arg_verbose)
    # Train the technical model
    if arg_model_type in ['technical', 'both']:
        tech_train_result_df, tech_test_df, tech_weight_file = train_technical_model(p_model_name=arg_model_name,
                                                                                     p_dataset_dir=arg_dataset_dir,
                                                                                     p_sample_size=arg_sample_size,
                                                                                     p_weight_path=arg_aes_weight_path,
                                                                                     p_batch_size=arg_batch_size,
                                                                                     p_metrics=arg_metrics,
                                                                                     p_epochs=arg_epochs,
                                                                                     p_verbose=arg_verbose)
