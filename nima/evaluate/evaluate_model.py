import argparse
import os

import pandas as pd
import scipy.stats as ss

from nima.config import DATASET_DIR, MODEL_BUILD_TYPE, print_msg, INPUT_SHAPE, TID_DATASET_DIR, \
    CROP_SHAPE, AVA_DATASET_DIR, RESULTS_DIR
from nima.model.data_generator import TestDataGenerator
from nima.model.loss import earth_movers_distance, mean_abs_percentage_ava, two_class_quality_acc, \
    mean_abs_percentage_acc
from nima.model.model_builder import NIMA, TechnicalModel, get_naming_prefix
from nima.utils.ava_dataset_utils import get_ava_csv_score_df
from nima.utils.preprocess import get_mean_quality_score, normalize_ratings
from nima.utils.tid_dataset_utils import get_mos_csv_df


def eval_aesthetic_model(p_model_name, p_dataset_dir=AVA_DATASET_DIR, p_sample_size=300,
                         p_weight_file=None, p_batch_size=32):
    from nima.utils.ava_dataset_utils import get_rating_columns
    assert os.path.isfile(p_weight_file), f"Invalid weights file {p_weight_file}"

    dataset_dir = AVA_DATASET_DIR if p_dataset_dir is None else p_dataset_dir
    images_dir = os.path.join(dataset_dir, 'images')
    img_format = 'jpg'
    print_msg(f'Images directory {images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', get_rating_columns()
    # keep_columns = ['image_id', 'max_rating', 'mean_score'] + get_rating_columns()
    # Get the DataFrame
    ava_csv_df = get_ava_csv_score_df(dataset_dir)  # Get the AVA csv dataframe
    # ava_csv_df['mean_score'] = ava_csv_df['mean_score'] / 9.0  # Normalize mean
    # check if samples size is given
    sample_size = p_sample_size
    if p_sample_size is None or p_sample_size > len(ava_csv_df):
        sample_size = len(ava_csv_df)
    # Load the test dataframe if not passed
    df_test = ava_csv_df.sample(n=sample_size).reset_index(drop=True)
    assert len(df_test) > 0, 'Empty dataframe'

    # form the model
    print_msg("Evaluating Model...")
    # Form the NIMA Aesthetic Model
    nima_aesthetic_cnn = NIMA(base_model_name=p_model_name, input_shape=INPUT_SHAPE,
                              crop_size=CROP_SHAPE, base_cnn_weight=None)
    nima_aesthetic_cnn.build()
    nima_aesthetic_cnn.model.load_weights(p_weight_file)
    nima_aesthetic_cnn.compile()

    # Get the generator
    test_batch_size = min(p_batch_size, 32, len(df_test))
    test_generator = TestDataGenerator(df_test, images_dir, x_col=x_col, y_col=None,
                                       img_format=img_format, num_classes=10,
                                       preprocess_input=nima_aesthetic_cnn.get_preprocess_function(),
                                       batch_size=p_batch_size, input_size=INPUT_SHAPE, )

    print_msg(f'Testing with Batch size:{test_batch_size}', 1)
    eval_result = nima_aesthetic_cnn.model.evaluate(test_generator)

    print_msg(f"loss({nima_aesthetic_cnn.loss}) : {eval_result[0]} | "
              f"accuracy({nima_aesthetic_cnn.metrics}) : {eval_result[1:]}", 1)
    # predict the values from model
    predictions = nima_aesthetic_cnn.model.predict(test_generator)
    print_msg(predictions.shape)

    # view the accuracy
    pred_columns = [f'pred_{column}' for column in y_cols]
    df_pred = pd.DataFrame(predictions, columns=pred_columns)
    df_pred['pred_mean_score'] = df_pred[pred_columns].apply(lambda x: get_mean_quality_score(normalize_ratings(x))
                                                             , axis=1)
    df_test = pd.concat([df_test, df_pred], axis=1)
    spearmanr = ss.spearmanr(df_test['mean_score'].to_numpy(), df_test['pred_mean_score'].to_numpy())[0]
    pearsonr = ss.pearsonr(df_test['mean_score'].to_numpy(), df_test['pred_mean_score'].to_numpy())[0]
    emd = earth_movers_distance(df_test['mean_score'], df_test['pred_mean_score'])
    two_class = two_class_quality_acc(df_test['mean_score'], df_test['pred_mean_score'])
    map = mean_abs_percentage_acc(df_test['mean_score'], df_test['pred_mean_score'])
    print_msg(f"spearman_correlation : {spearmanr},"
              f" pearson_correlation : {pearsonr}"
              f" Earth Movers Distance : {emd}"
              f" Two class Quality : {two_class}"
              f" Mean Absolute Percentage : {map}"
              , 1)
    return df_test[['image_id', 'mean_score', 'pred_mean_score']]


def eval_technical_model(p_model_name, p_dataset_dir=TID_DATASET_DIR, p_sample_size=300,
                         p_weight_file=None, p_batch_size=32):
    """
    Trains an technical model for the given parameters.
    :param p_weight_file:
    :param p_model_name: Model for evaluation.
    :param p_dataset_dir: TID dataset path, used when test_df is not passed.
    :param p_sample_size: No. of samples to test if DataFrame not given.
    :param p_batch_size: Test batch size
    :return:
    """
    assert os.path.isfile(p_weight_file), f"Invalid weights file {p_weight_file}"
    tid_dataset_dir = TID_DATASET_DIR if p_dataset_dir is None else p_dataset_dir
    tid_images_dir = os.path.join(tid_dataset_dir, 'distorted_images')
    print_msg(f'Images directory {tid_images_dir}')

    x_col, y_cols = 'image_id', 'mean'
    img_format = 'bmp'

    # Get the DataFrame
    tid_df = get_mos_csv_df(tid_dataset_dir)
    tid_df['mean'] = tid_df['mean']
    # check if samples size is given
    sample_size = p_sample_size
    if p_sample_size is None or p_sample_size > len(tid_df):
        sample_size = len(tid_df)
    # Load the test dataframe if not passed
    df_test = tid_df.sample(n=sample_size).reset_index(drop=True)
    assert len(df_test) > 0, 'Empty dataframe'

    # Load the model
    print_msg("Evaluating Model...")
    # Form the NIMA Aesthetic Model
    nima_tech_cnn = TechnicalModel(model_name=p_model_name, base_cnn_weight=None,
                                   input_shape=INPUT_SHAPE, crop_size=CROP_SHAPE, )

    train_weights_file = p_weight_file

    nima_tech_cnn.build()
    print_msg(f'Loading weights {train_weights_file}', 1)
    nima_tech_cnn.model.load_weights(train_weights_file)
    nima_tech_cnn.compile()

    # Get the generator
    test_batch_size = min(p_batch_size, 32, len(df_test))
    test_generator = TestDataGenerator(df_test, tid_images_dir, x_col=x_col, y_col=None,
                                       img_format=img_format, num_classes=1,
                                       preprocess_input=nima_tech_cnn.get_preprocess_function(),
                                       batch_size=test_batch_size, input_size=INPUT_SHAPE)

    # Get the prediction
    print_msg(f'Testing with Batch size:{test_batch_size}', 1)
    # evaluate model
    eval_result = nima_tech_cnn.model.evaluate(test_generator)

    print_msg(f"loss({nima_tech_cnn.loss}) : {eval_result[0]} | "
              f"accuracy({nima_tech_cnn.metrics}) : {eval_result[1:]}", 1)
    # predict the values from model
    predictions = nima_tech_cnn.model.predict(test_generator)
    df_test['pred_mean'] = predictions

    # view the accuracy
    spearmanr = ss.spearmanr(df_test['mean'].to_numpy(), df_test['pred_mean'].to_numpy())[0]
    pearsonr = ss.pearsonr(df_test['mean'].to_numpy(), df_test['pred_mean'].to_numpy())[0]

    predict_df_filename = get_naming_prefix(nima_tech_cnn.model_type, nima_tech_cnn.model_class_name,
                                            prefix='eval') + '_pred.csv'
    predict_file = os.path.join(RESULTS_DIR, predict_df_filename)
    print_msg(f'saving predictions to {predict_file}', 1)
    df_test.to_csv(predict_file, index=False)
    return pearsonr, spearmanr, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model on AVA Dataset')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=True,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-t', '--model-type', type=str, default=MODEL_BUILD_TYPE[0], required=True,
                        help="Model type to train aesthetic/technical.")
    parser.add_argument('-d', '--dataset-dir', type=str, default=DATASET_DIR, required=True,
                        help='Dataset directory.')
    parser.add_argument('-w', '--weight-file', type=str, required=True,
                        help="Model path to evaluate.")
    parser.add_argument('-s', '--sample-size', type=int, default=300, required=False,
                        help='No. of samples to be picked at random for evaluation, Default is 30.')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=32, required=False, help='Test Batch size.')
    args = parser.parse_args()

    # Set the Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'

    # model to choose, default to mobilenet
    arg_model_name = args.__dict__['model_name']
    arg_model_type = args.__dict__['model_type']
    assert arg_model_type is not None and arg_model_type in MODEL_BUILD_TYPE, f'Model type should have value {MODEL_BUILD_TYPE}'

    arg_batch_size = args.__dict__['batch_size']
    arg_sample_size = args.__dict__['sample_size']
    assert arg_sample_size is not None, 'Sample size must be between 30 and 1000'

    arg_weight_file = args.__dict__['weight_file']
    assert os.path.isfile(arg_weight_file), f'Invalid weight file {arg_weight_file}, does not exists.'

    # Evaluate the aesthetic model
    if arg_model_type in [MODEL_BUILD_TYPE[0]]:
        aes_train_result_df, aes_test_df, aes_weight_file = eval_aesthetic_model(p_model_name=arg_model_name,
                                                                                 p_dataset_dir=arg_dataset_dir,
                                                                                 p_sample_size=arg_sample_size,
                                                                                 p_batch_size=arg_batch_size,
                                                                                 p_weight_file=arg_weight_file)
    # Evaluate the technical model
    if arg_model_type in [MODEL_BUILD_TYPE[1]]:
        tech_train_result_df = eval_technical_model(p_model_name=arg_model_name,
                                                    p_dataset_dir=arg_dataset_dir,
                                                    p_sample_size=arg_sample_size,
                                                    p_batch_size=arg_batch_size,
                                                    p_weight_file=arg_weight_file)
