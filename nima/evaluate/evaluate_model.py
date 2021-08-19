import argparse
import os

from nima.config import DATASET_DIR, MODEL_BUILD_TYPE, print_msg, INPUT_SHAPE, TID_DATASET_DIR, \
    CROP_SHAPE, AVA_DATASET_DIR
from nima.model.data_generator import TestDataGenerator
from nima.model.loss import earth_movers_distance, pearson_corelation, spearman_corelation
from nima.model.model_builder import NIMA, TechnicalModel
from nima.utils.ava_dataset_utils import get_ava_csv_score_df
from nima.utils.tid_dataset_utils import get_mos_csv_df


def test_aesthetic_model(p_model_name, p_dataset_dir=TID_DATASET_DIR, p_sample_size=300,
                         p_weight_file=None, p_batch_size=32):
    from nima.utils.ava_dataset_utils import get_rating_columns
    assert os.path.isfile(p_weight_file), f"Invalid weights file {p_weight_file}"

    dataset_dir = AVA_DATASET_DIR if p_dataset_dir is None else p_dataset_dir
    images_dir = os.path.join(dataset_dir, 'images')
    img_format = 'jpg'
    print_msg(f'Images directory {images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', get_rating_columns()
    keep_columns = ['image_id', 'max_rating', 'mean_score'] + get_rating_columns()
    # Get the DataFrame
    ava_csv_df = get_ava_csv_score_df(dataset_dir)  # Get the AVA csv dataframe
    ava_csv_df['mean_score'] = ava_csv_df['mean_score'] / 9.0  # Normalize mean
    # check if samples size is given
    sample_size = p_sample_size
    if p_sample_size is None or p_sample_size > len(ava_csv_df):
        sample_size = len(ava_csv_df)
    # Load the test dataframe if not passed
    df_test = ava_csv_df.sample(n=sample_size).reset_index(drop=True)
    assert len(df_test) > 0, 'Empty dataframe'

    # form the model
    print_msg("Testing Model...")
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
    eval_result, df_test = nima_aesthetic_cnn.evaluate_model(df_test, test_generator, prefix='freezed')

    # view the accuracy
    pred_columns = [column for column in df_test.columns.tolist() if column.startswith('pred')]
    emd = earth_movers_distance(df_test[get_rating_columns()],
                                df_test[pred_columns].drop(['pred_mean_score', 'pred_max_rating'], axis=1))
    print_msg(f"For mean score - mse : {spearman_correlation}, mae : {pearson_correlation}")
    print_msg(f"For ratings - emd : {emd}")
    print_msg(df_test.iloc[0])
    return spearman_correlation, pearson_correlation, df_test


def eval_technical_model(p_model_name, p_dataset_dir=TID_DATASET_DIR, p_sample_size=300,
                         p_weight_file=None, p_batch_size=32):
    """
    Trains an aesthetic model for the given parameters.
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
    eval_result, df_test = nima_tech_cnn.evaluate_model(df_test, test_generator)

    # view the accuracy
    spearman_correlation = spearman_corelation(df_test['mean'].to_numpy(), df_test['pred_mean'].to_numpy())
    pearson_correlation = pearson_corelation(df_test['mean'].to_numpy(), df_test['pred_mean'].to_numpy())
    print_msg(f"pearson_correlation : {pearson_correlation}, spearman_correlation : {spearman_correlation}")
    print_msg(df_test.iloc[0])
    return pearson_correlation, spearman_correlation, df_test


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
        aes_train_result_df, aes_test_df, aes_weight_file = test_aesthetic_model(p_model_name=arg_model_name,
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
