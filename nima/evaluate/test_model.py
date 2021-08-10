import argparse
import os

from keras.losses import mean_squared_error

from nima.config import DATASET_DIR, MODEL_BUILD_TYPE, print_msg, INPUT_SHAPE, PROJECT_ROOT_DIR, WEIGHTS_DIR
from nima.model.data_generator import TestDataGenerator
from nima.model.loss import earth_movers_distance
from nima.model.model_builder import NIMA
from nima.utils.ava_dataset_utils import get_ava_csv_df


def test_aesthetic_model(p_model_name, p_dataset_dir, p_sample_size, p_weight_path,
                         p_freeze_base_model, p_batch_size, p_metrics, p_verbose):
    from nima.utils.ava_dataset_utils import load_data, get_rating_columns
    ava_dataset_dir = os.path.join(p_dataset_dir, 'AVA')
    ava_images_dir = os.path.join(ava_dataset_dir, 'images')
    img_format = 'jpg'
    print_msg(f'Images directory {ava_images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', get_rating_columns()
    df_test = get_ava_csv_df(ava_dataset_dir).sample(p_sample_size)
    assert len(df_test) > 0, 'Empty dataframe'

    test_batch_size = min(p_batch_size, 32, len(df_test))
    # form the model
    print_msg("Testing Model...")
    nima_aes_cnn = NIMA(base_model_name=p_model_name, weights=p_weight_path, model_type='aesthetic',
                        loss=earth_movers_distance, input_shape=INPUT_SHAPE, metrics=p_metrics,
                        freeze_base_model=p_freeze_base_model, )
    nima_aes_cnn.build()
    nima_aes_cnn.compile()
    # Get the generator
    test_generator = TestDataGenerator(df_test, ava_images_dir, x_col=x_col, y_col=None,
                                       img_format=img_format, num_classes=10,
                                       preprocess_input=nima_aes_cnn.get_preprocess_function(),
                                       batch_size=test_batch_size, input_size=INPUT_SHAPE,
                                       )

    train_weights_file = nima_aes_cnn.get_naming_prefix() + '.hdf5'
    assert os.path.isfile(train_weights_file), 'Weights does not exist, please train the model first'
    print_msg(f'Testing Batch size:{test_batch_size}', 1)
    # Get the prediction
    predictions = nima_aes_cnn.model.predict(test_generator, use_multiprocessing=True, verbose=p_verbose)
    df_test['rating_predict'] = predictions
    print(df_test.iloc[0:5])
    # Save predictions to csv
    predict_df_filename = nima_aes_cnn.get_naming_prefix() + '_pred.csv'
    predict_file = os.path.join(PROJECT_ROOT_DIR, 'nima', 'evaluate', predict_df_filename)
    df_test.to_csv(predict_file, index=False)
    return df_test


def test_technical_model(p_model_name, p_dataset_dir, p_sample_size, p_freeze_base_model,
                         p_batch_size, p_metrics, p_verbose):
    """
    Trains an aesthetic model for the given parameters.
    :param p_model_name:
    :param p_dataset_dir:
    :param p_sample_size:
    :param p_batch_size: Test batch size
    :param p_metrics: use ['mean_absolute_error'] for regression
    :param p_verbose: 0 or 1
    :return:
    """
    from nima.utils.tid_dataset_utils import get_mos_df
    tid_dataset_dir = os.path.join(p_dataset_dir, 'tid2013')
    tid_images_dir = os.path.join(tid_dataset_dir, 'distorted_images')
    img_format = 'bmp'
    print_msg(f'Images directory {tid_images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', 'rating'
    df_test = get_mos_df(tid_dataset_dir).sample(n=p_sample_size).reset_index(drop=True)
    assert len(df_test) > 0, 'Empty dataframe'

    test_batch_size = min(p_batch_size, 32, len(df_test))
    # form the model
    print_msg("Testing Model...")
    nima_tech_cnn = NIMA(base_model_name=p_model_name, model_type='technical',
                         loss=mean_squared_error, input_shape=INPUT_SHAPE, metrics=p_metrics,
                         freeze_base_model=p_freeze_base_model)
    train_weights_file = os.path.join(WEIGHTS_DIR, nima_tech_cnn.get_naming_prefix() + '.hdf5')
    nima_tech_cnn.build()
    nima_tech_cnn.compile()
    print_msg(f'Loading weights {train_weights_file}', 1)
    nima_tech_cnn.model.load_weights(train_weights_file)
    # Get the generator
    test_generator = TestDataGenerator(df_test, tid_images_dir, x_col=x_col, y_col=None,
                                       img_format=img_format, num_classes=1,
                                       preprocess_input=nima_tech_cnn.get_preprocess_function(),
                                       batch_size=test_batch_size, input_size=INPUT_SHAPE)

    assert os.path.isfile(train_weights_file), 'Weights does not exist, please train the model first'
    # Get the prediction
    print_msg(f'Testing Batch size:{test_batch_size}', 1)
    predictions = nima_tech_cnn.model.predict(test_generator, use_multiprocessing=True, verbose=p_verbose)
    print_msg(f'length {len(predictions)}', 2)
    df_test['y_pred'] = predictions
    print(df_test.to_numpy()[0:5])
    # Save predictions to csv
    predict_df_filename = nima_tech_cnn.get_naming_prefix() + '_pred.csv'
    predict_file = os.path.join(PROJECT_ROOT_DIR, 'nima', 'evaluate', predict_df_filename)
    df_test.to_csv(predict_file, index=False)
    return df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model on AVA Dataset')
    parser.add_argument('-d', '--dataset-dir', type=str, default=DATASET_DIR, required=True,
                        help='Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=True,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-t', '--model-type', type=str, default=MODEL_BUILD_TYPE[0], required=True,
                        help="Model type to train aesthetic/technical/both.")
    parser.add_argument('-w', '--weights-dir', type=str, default=WEIGHTS_DIR, required=True,
                        help="Model type to train aesthetic/technical/both.")
    parser.add_argument('-s', '--sample-size', type=int, default=100, required=False,
                        help='No. of random evaluate samples, Default is 30.')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=32, required=False, help='Test Batch size.')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        required=False, help='Verbose, default 0.')
    args = parser.parse_args()

    # Set the Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'

    # model to choose, default to mobilenet
    arg_model_name = args.__dict__['model_name']
    arg_model_type = args.__dict__['model_type']
    arg_batch_size = args.__dict__['batch_size']
    arg_sample_size = args.__dict__['sample_size']
    assert arg_sample_size is not None, 'Sample size must be between 30 and 1000'
    arg_verbose = args.__dict__['verbose']
    arg_freeze_base = args.__dict__['freeze_base']

    # Train the aesthetic model
    if arg_model_type in [MODEL_BUILD_TYPE[0], 'both']:
        aes_train_result_df, aes_test_df, aes_weight_file = test_aesthetic_model(p_model_name=arg_model_name,
                                                                                 p_dataset_dir=arg_dataset_dir,
                                                                                 p_sample_size=arg_sample_size,
                                                                                 p_freeze_base_model=arg_freeze_base,
                                                                                 p_batch_size=arg_batch_size,
                                                                                 p_metrics=['accuracy'],
                                                                                 p_verbose=arg_verbose)
    # Train the technical model
    if arg_model_type in [MODEL_BUILD_TYPE[1], 'both']:
        tech_train_result_df, tech_test_df, tech_weight_file = test_technical_model(p_model_name=arg_model_name,
                                                                                    p_dataset_dir=arg_dataset_dir,
                                                                                    p_sample_size=arg_sample_size,
                                                                                    p_freeze_base_model=arg_freeze_base,
                                                                                    p_batch_size=arg_batch_size,
                                                                                    p_metrics=['mean_absolute_error'],
                                                                                    p_verbose=arg_verbose)
