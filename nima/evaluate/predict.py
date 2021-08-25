import argparse
import os

import pandas as pd

from nima.config import INPUT_SHAPE, print_msg, CROP_SHAPE, RESULTS_DIR
from nima.model.data_generator import TestDataGenerator
from nima.model.model_builder import TechnicalModel, NIMA
from nima.utils.ava_dataset_utils import get_rating_columns
from nima.utils.image_utils import get_images
from nima.utils.preprocess import show_images_with_score, get_mean_quality_score, normalize_ratings


def get_aesthetic_prediction(p_model_name, p_dataset_dir, p_weight_file, df):
    print_msg('Aesthetic Model ....')

    # Load the model
    train_weights_file = p_weight_file
    nima_aes_cnn = NIMA(base_model_name=p_model_name, base_cnn_weight=None,
                        input_shape=INPUT_SHAPE, crop_size=CROP_SHAPE, )
    nima_aes_cnn.build()
    # print_msg(f'Loading weights {train_weights_file}', 1)
    nima_aes_cnn.model.load_weights(train_weights_file)
    nima_aes_cnn.compile()

    # Get the generator
    img_format = 'jpg'
    test_generator = TestDataGenerator(df, p_dataset_dir, x_col='image_id', y_col=None,
                                       img_format=img_format, num_classes=10,
                                       preprocess_input=nima_aes_cnn.get_preprocess_function(),
                                       batch_size=32, input_size=INPUT_SHAPE)

    # Get the prediction
    predictions = nima_aes_cnn.model.predict(test_generator)
    pred_columns = [f'pred_{column}' for column in get_rating_columns()]
    df_pred = pd.DataFrame(predictions, columns=pred_columns)
    df_pred['aes_mean_score'] = df_pred[pred_columns].apply(lambda x:
                                                             get_mean_quality_score(normalize_ratings(x)), axis=1)
    return pd.concat([df, df_pred], axis=1)


def get_technical_predictions(p_model_name, p_dataset_dir, p_weight_file, df):
    print_msg('Technical Model ....')

    # Load the model
    # print_msg("Creating Technical Model...")
    train_weights_file = p_weight_file
    nima_tech_cnn = TechnicalModel(model_name=p_model_name, base_cnn_weight=None,
                                   input_shape=INPUT_SHAPE, crop_size=CROP_SHAPE, )
    nima_tech_cnn.build()
    # print_msg(f'Loading weights {train_weights_file}', 1)
    nima_tech_cnn.model.load_weights(train_weights_file)
    nima_tech_cnn.compile()

    # Get the generator
    img_format = 'jpg'
    test_generator = TestDataGenerator(df, p_dataset_dir, x_col='image_id', y_col=None,
                                       img_format=img_format, num_classes=1,
                                       preprocess_input=nima_tech_cnn.get_preprocess_function(),
                                       batch_size=32, input_size=INPUT_SHAPE)

    # Get the prediction
    predictions = nima_tech_cnn.model.predict(test_generator)
    df['tech_mean_score'] = predictions
    return df


def main(aes_model_name, tech_model_name, img_dataset_dir, aes_weight_file, tech_weight_file):
    print_msg(f'Images directory {images_dir}')
    # rename the images to id format, if required
    # rename_images(images_dir)
    # Check the images present in the given directory
    images = get_images(img_dataset_dir)
    images = [os.path.basename(file).replace('.jpg', '') for file in images]  # remove path from filename

    x_col, y_cols = 'image_id', 'mean'
    # Load the test dataframe if not passed
    assert len(images) > 0, 'Empty directory, no images found.'
    test_df = pd.DataFrame(images, columns=['image_id'])
    test_df['image_id'] = test_df['image_id'].astype('int')

    # Get the model prediction
    tech_predict_df = get_technical_predictions(p_model_name=tech_model_name, p_dataset_dir=img_dataset_dir,
                                                p_weight_file=tech_weight_file, df=test_df.copy())
    print_msg("")
    aes_predict_df = get_aesthetic_prediction(p_model_name=aes_model_name, p_dataset_dir=img_dataset_dir,
                                              p_weight_file=aes_weight_file, df=test_df.copy())
    print_msg("")
    predict_df = pd.merge(tech_predict_df, aes_predict_df, on='image_id')
    file_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    print_msg(f'Saved predictions to {file_path}')
    predict_df.to_csv(file_path, index=False)
    # Display the images
    show_images_with_score(predict_df, images_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict the image quality')
    parser.add_argument('-d', '--dataset-dir', type=str, default=None, required=True,
                        help='Image Dataset directory having images to test/Predict.')
    parser.add_argument('-n', '--aes-model-name', type=str, default='mobilenet', required=False,
                        help="Aesthetic Base model name")
    parser.add_argument('-nt', '--tech-model-name', type=str, default='mobilenet_v2', required=False,
                        help="Technical Base model name")
    parser.add_argument('-tp', '--aesthetic-model-path', type=str,
                        default='../../weights/technical/weights-mobilenetv2-technical.hdf5', required=True,
                        help="Model type to train aesthetic/technical.")
    parser.add_argument('-ap', '--technical-model-path', type=str,
                        default='../../weights/aesthetic/weights-mobilenet-aesthetic.hdf5', required=True,
                        help="Model Name to use for prediction.\n"
                             "view models.json to know available models for prediction.")
    args = parser.parse_args()

    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'
    images_dir = arg_dataset_dir
    # Model name
    aes_model_name = args.__dict__['aes_model_name']
    tech_model_name = args.__dict__['tech_model_name']
    # model path
    aes_weight_file = args.__dict__['aesthetic_model_path']
    assert os.path.isfile(aes_weight_file), f'Invalid model path, weight does not exists : {aes_weight_file}.'
    tech_weight_file = args.__dict__['aesthetic_model_path']
    assert os.path.isfile(aes_weight_file), f'Invalid model path, weight does not exists : {aes_weight_file}.'

    # Call the main method
    main(aes_model_name, tech_model_name, arg_dataset_dir, aes_weight_file, tech_weight_file)
