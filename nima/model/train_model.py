import argparse
import os
from pathlib import Path

import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback
from sklearn.model_selection import train_test_split

from nima.model.data_generator import NimaDataGenerator
from nima.model.model_builder import NIMA
from nima.utils.ava_preprocess import get_ava_csv_df
from nima.utils.ava_preprocess import get_rating_columns

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'nima', 'weights')
AVA_DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')
AVA_IMAGES_DIR = ""


def load_data():
    """
    Returns the pandas DataFrame for Training Data, Test Data and Validation Data.
    :return: Train DataFrame, Validation DataFrame, Test DataFrame
    """
    ava_csv_df = get_ava_csv_df(AVA_IMAGES_DIR)  # Get the AVA csv dataframe
    count_columns = get_rating_columns()  # get the columns representing ratings
    keep_columns = ['image_id'] + count_columns

    df_train, df_test = train_test_split(df, test_size=0.05, shuffle=True, random_state=1024)
    df_train, df_valid = train_test_split(df_train, test_size=0.3, shuffle=True, random_state=1024)

    return df_train, df_valid, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on AVA Dataset')
    parser.add_argument('-d', '--dataset-dir', type=str, required=False, help='AVA Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=False,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-m', '--metrics', type=list, default=['accuracy'], required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-w', '--weights-path', type=str, default=None, required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-b', '--batch-size', type=int, default=64, required=False, help='Batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=15, required=False, help='Number of epochs, default 10.')
    parser.add_argument('-v', '--verbose', type=int, default=0, required=False, help='Verbose, default 0.')
    args = parser.parse_args()

    # Set the AVA Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    if arg_dataset_dir is not None:
        AVA_DATASET_DIR = arg_dataset_dir
    AVA_IMAGES_DIR = os.path.join(AVA_DATASET_DIR, 'images')

    # model to choose, default to mobilenet
    arg_model_name = args.__dict__['model_name']
    arg_weight_path = args.__dict__['weights_path']
    if arg_weight_path is not None:
        assert os.path.isfile(arg_weight_path), 'Invalid weights, does not exists.'

    arg_batch_size = args.__dict__['batch_size']
    arg_epochs = args.__dict__['epochs']
    arg_verbose = args.__dict__['verbose']
    arg_metrics = args.__dict__['metrics']

    # ava_csv_df = get_ava_csv_df()  # Get the AVA csv dataframe
    # count_columns = get_rating_columns()  # get the columns representing ratings
    # keep_columns = ['image_id'] + count_columns
    # df = ava_csv_df.drop(keep_columns)
    # df = load_data()
    # Get the test size and test size
    # df_train, df_test = train_test_split(df, test_size=0.05, shuffle=True, random_state=1024)
    # df_train, df_valid = train_test_split(df_train, test_size=0.3, shuffle=True, random_state=1024)
    df_train, df_valid, df_test = load_data()
    # Form the NIMA Model
    nima_model = NIMA(base_model_name=arg_model_name, weights='imagenet', input_shape=(224, 224, 3),
                      metrics=arg_metrics)
    nima_model.build()
    # load model weights if existing
    if arg_weight_path is not None:
        nima_model.model.load_weights(arg_weight_path)

    # Get the generator
    train_generator = NimaDataGenerator()
    valid_generator = NimaDataGenerator()

    # set model weight and path
    weight_filename = f'{arg_model_name}_weight_best'
    weight_filepath = os.path.join(WEIGHTS_DIR, weight_filename)
    print(f'Model Weight path : {weight_filepath}')

    es = EarlyStopping(monitor='val_loss', patience=4, verbose=arg_verbose)
    ckpt = ModelCheckpoint(
        filepath=weight_filepath,
        save_weights_only=True,
        monitor="val_f2_score",
        mode="auto",
        save_best_only=True,
    )
    lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    plot_loss = PlotLossesCallback()

    # start training
    history = nima_model.model.fit(train_generator, validation_data=valid_generator,
                                   epochs=arg_epochs, callbacks=[es, ckpt, lr, plot_loss],
                                   verbose=arg_verbose)
    result_df = pd.DataFrame(history.history)
    preprocess_input = nima_model.preprocess_input()

    nima_model.compile()
    nima_model.fit()
