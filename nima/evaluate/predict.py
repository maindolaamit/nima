import argparse
import os
from glob import glob

import pandas as pd

from nima.config import WEIGHTS_DIR, INPUT_SHAPE
from nima.model.data_generator import TrainDataGenerator
from nima.model.model_builder import NIMA


def get_images_df(images_dir, img_format):
    image_files = [os.path.basename(name) for name in glob(os.path.join(images_dir, f'*.{img_format}'))]
    df_image = pd.DataFrame(image_files, columns=['image_id'])
    return df_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict the image quality')
    parser.add_argument('-d', '--dataset-dir', type=str, default=None, required=False,
                        help='Image Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=False,
                        help='Model Name to use for prediction, view models.json to know available models for prediction.')
    args = parser.parse_args()

    # Set the AVA Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(
        arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'
    images_dir = arg_dataset_dir

    # model to choose, default to mobilenet
    arg_model_name = args.__dict__['model_name']
    weight_file = os.path.join(WEIGHTS_DIR, f'{arg_model_name}.hdf5')
    assert os.path.isfile(weight_file), f'Invalid model name, weight does not exists : {weight_file}.'

    # Load the dataset
    df_test = get_images_df(images_dir)
    # Form the NIMA Model
    nima_cnn = NIMA(base_model_name=arg_model_name, input_shape=INPUT_SHAPE,
                    metrics=['accuracy'])
    nima_cnn.build()
    # load model weights
    weight_filepath = nima_cnn.get_weight_path()
    nima_cnn.model.load_weights(weight_file)
    nima_cnn.compile()

    # Get the generator
    train_generator = TrainDataGenerator(df_test, images_dir, x_col='image_id', y_col=None,
                                         preprocess_input=nima_cnn.get_preprocess_function(),
                                         is_train=False, batch_size=32, )

    # predict from model
    print("Training Model...")
    predictions = nima_cnn.model.predict(train_generator)
    print(predictions)
