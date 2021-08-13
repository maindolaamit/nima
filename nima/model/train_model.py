import argparse
import os
import time

import pandas as pd
# Disable extra logs
import tensorflow as tf
from keras.losses import mean_squared_error

from nima.config import INPUT_SHAPE, DATASET_DIR, CROP_SHAPE, MODEL_BUILD_TYPE, print_msg, WEIGHTS_DIR, TID_DATASET_DIR
from nima.evaluate.evaluate_model import eval_technical_model
from nima.model.data_generator import TrainDataGenerator, TestDataGenerator
from nima.model.loss import earth_movers_distance
from nima.model.model_builder import NIMA, model_weight_name

tf.get_logger().setLevel('ERROR')  # Limit the tensorflow logs to ERROR only


def train_model_cv(model_name, model_type, images_dir,
                   df, x_col, y_cols, img_format, metrics, loss=earth_movers_distance,
                   batch_size=64, epochs=32, verbose=0, freeze_base_model=False):
    """
    Train the final model for the given number of parameters
    :param model_name:
    :param model_type:
    :param images_dir: Images directory
    :param df:
    :param x_col: x_col in the DataFrame
    :param y_cols: y_col in the DataFrame
    :param img_format: Image format in the directory, to be appended to the image_id
    :param loss:
    :param metrics:
    :param batch_size: Train batch_size
    :param epochs: Number of epochs
    :param verbose: verbose
    :param freeze_base_model:
    :return: Training result_df and saved models weight path
    """
    from sklearn.model_selection import KFold
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
    from livelossplot import PlotLossesKerasTF
    # create a cross fold
    cv = KFold(n_splits=5, shuffle=True, random_state=1024)
    fold = 1
    results_df_folds = acc_folds = loss_folds = []
    es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=verbose)
    lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=verbose)
    # Loop for each fold
    for train_index, val_index in cv.split(df[x_col]):
        df_train, df_val = df.iloc[train_index], df.iloc[val_index]
        # create the model
        nima_cnn = NIMA(base_model_name=model_name, model_type=model_type,
                        input_shape=INPUT_SHAPE, metrics=metrics)
        nima_cnn.build()
        nima_cnn.compile()

        # Form the model callbacks
        model_save_name = f"{model_weight_name(model_name, model_type, freeze_base_model)}_fold{fold}"
        # Live loss
        liveloss_filename = os.path.join(WEIGHTS_DIR, model_save_name)
        from livelossplot.outputs import MatplotlibPlot
        plot_loss = PlotLossesKerasTF(outputs=[MatplotlibPlot(figpath=liveloss_filename)])
        # Model checkpoint
        weight_filepath = os.path.join(WEIGHTS_DIR, model_save_name)
        ckpt = ModelCheckpoint(
            filepath=weight_filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
        )

        print_msg(f'Model Weight path : {weight_filepath}', 1)
        csv_log = CSVLogger(f"{model_save_name}.csv")
        callbacks = [es, ckpt, lr, csv_log, plot_loss]

        # Get the generator
        train_generator = TrainDataGenerator(df_train, images_dir, x_col=x_col, y_col=y_cols,
                                             input_size=INPUT_SHAPE, crop_size=CROP_SHAPE,
                                             img_format=img_format, num_classes=1,
                                             preprocess_input=nima_cnn.get_preprocess_function(),
                                             batch_size=batch_size)
        valid_generator = TrainDataGenerator(df_val, images_dir, x_col, y_cols,
                                             input_size=INPUT_SHAPE, crop_size=CROP_SHAPE,
                                             img_format=img_format, num_classes=1,
                                             preprocess_input=nima_cnn.get_preprocess_function(),
                                             batch_size=batch_size)

        # start training
        start_time = time.perf_counter()
        history = nima_cnn.model.fit(train_generator, validation_data=valid_generator,
                                     epochs=epochs, callbacks=callbacks,
                                     verbose=verbose, use_multiprocessing=True)
        end_time = time.perf_counter()
        print_msg(f'Training Time (HH:MM:SS) : {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}', 1)
        result_df = pd.DataFrame(history.history)
        result_df['fold'] = fold
        results_df_folds.append(result_df)
        # Evaluate model
        loss, acc = nima_cnn.model.evaluate(train_generator)
        acc_folds.append(acc)
        loss_folds.append(loss)

        fold += 1
    return pd.concat(results_df_folds)


def train_aesthetic_model(p_model_name, p_dataset_dir, p_sample_size, p_weight_path,
                          p_batch_size, p_metrics, p_epochs, p_verbose):
    from nima.utils.ava_dataset_utils import load_data, get_rating_columns
    ava_dataset_dir = p_dataset_dir #os.path.join(p_dataset_dir, 'AVA')
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
    nima_aesthetic_cnn = NIMA(base_model_name=p_model_name, model_weights=p_weight_path,
                              loss=earth_movers_distance, input_shape=INPUT_SHAPE, metrics=p_metrics)

    # Build the model for training
    nima_aesthetic_cnn.build()
    nima_aesthetic_cnn.compile()
    nima_aesthetic_cnn.model.summary()

    # Get the generator
    train_generator = TrainDataGenerator(df_train, ava_images_dir, x_col=x_col, y_col=y_cols,
                                         img_format=img_format, num_classes=10,
                                         preprocess_input=nima_aesthetic_cnn.get_preprocess_function(),
                                         batch_size=train_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)
    valid_generator = TrainDataGenerator(df_valid, ava_images_dir, x_col=x_col, y_col=y_cols,
                                         img_format=img_format, num_classes=10,
                                         preprocess_input=nima_aesthetic_cnn.get_preprocess_function(),
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
                                       preprocess_input=nima_aesthetic_cnn.get_preprocess_function(),
                                       input_size=INPUT_SHAPE, batch_size=test_batch_size)

    predictions = nima_aesthetic_cnn.model.predict(test_generator)
    df_test['predictions'] = predictions
    return train_result_df, df_test, train_weights_file


def train_technical_model(p_model_name, p_dataset_dir, p_sample_size, p_loss,
                          p_batch_size, p_metrics, p_epochs, p_verbose):
    """
    Trains an aesthetic model for the given parameters.
    :param p_model_name:
    :param p_dataset_dir:
    :param p_sample_size:
    :param p_weight_path:
    :param p_batch_size:
    :param p_metrics: use ['mean_absolute_error'] for regression
    :param p_epochs:
    :param p_verbose:
    :return:
    """
    from nima.utils.tid_dataset_utils import load_tid_data
    tid_dataset_dir = TID_DATASET_DIR if p_dataset_dir is None else p_dataset_dir
    tid_images_dir = os.path.join(tid_dataset_dir, 'distorted_images')
    img_format = 'bmp'
    print_msg(f'Images directory {tid_images_dir}')

    # Load the dataset
    x_col, y_cols = 'image_id', 'rating'
    df_train, df_valid, df_test = load_tid_data(tid_dataset_dir, p_sample_size)
    assert len(df_train) > 0 and len(df_valid) > 0 and len(df_test) > 0, 'Empty dataframe'
    train_batch_size = valid_batch_size = p_batch_size
    test_batch_size = min(p_batch_size, 32, len(df_test))

    # Form the NIMA Aesthetic Model
    nima_tech_cnn = NIMA(base_model_name=p_model_name, model_weights='imagenet', model_type='technical',
                         loss=p_loss, input_shape=INPUT_SHAPE, metrics=p_metrics, )

    # Build the model for training
    nima_tech_cnn.build()
    nima_tech_cnn.compile()
    nima_tech_cnn.model.summary()

    # Get the generator
    train_generator = TrainDataGenerator(df_train, tid_images_dir, x_col=x_col, y_col=y_cols,
                                         img_format=img_format, num_classes=1,
                                         preprocess_input=nima_tech_cnn.get_preprocess_function(),
                                         batch_size=train_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)
    valid_generator = TrainDataGenerator(df_valid, tid_images_dir, x_col, y_cols, img_format=img_format, num_classes=1,
                                         preprocess_input=nima_tech_cnn.get_preprocess_function(),
                                         batch_size=valid_batch_size, input_size=INPUT_SHAPE, crop_size=CROP_SHAPE)

    # Train the model
    print_msg("Training Technical Model...")
    print_msg(f'Training Batch size {train_batch_size}, metric : {p_metrics}', 1)
    train_result_df = nima_tech_cnn.train_model(train_generator, valid_generator, epochs=p_epochs,
                                                verbose=p_verbose)

    # Test the model
    print_msg("Testing Model...")
    df_test = eval_technical_model(p_model_name=p_model_name, p_test_df=df_test,
                                   p_loss=p_loss, p_metrics=['mean_absolute_error'],
                                   p_batch_size=test_batch_size, p_verbose=p_verbose)
    # Get the generator
    # test_generator = TestDataGenerator(df_test, tid_images_dir, x_col=x_col, y_col=None,
    #                                    img_format=img_format, num_classes=1,
    #                                    preprocess_input=nima_tech_cnn.get_preprocess_function(),
    #                                    batch_size=test_batch_size, input_size=INPUT_SHAPE)
    #
    # train_weights_file = nima_tech_cnn.get_naming_prefix() + '.hdf5'
    # predict_df_filename = nima_tech_cnn.get_naming_prefix() + '_pred.csv'
    #
    # nima_tech_cnn_test = NIMA(base_model_name=p_model_name, weights=train_weights_file, model_type='technical',
    #                           loss=mean_squared_error, input_shape=INPUT_SHAPE, metrics=p_metrics,
    #                           freeze_base_model=p_freeze_base)
    # nima_tech_cnn_test.build()
    # nima_tech_cnn_test.compile()
    # # test_steps = np.ceil(len(test_generator) / test_batch_size)
    # print_msg(f'Testing Batch size:{test_batch_size}', 1)
    # predictions = nima_tech_cnn_test.model.predict(test_generator, use_multiprocessing=True, verbose=p_verbose)
    # df_test['rating_predict'] = predictions
    # df_test.to_csv(predict_df_filename, index=False)
    # print(df_test.iloc[0])
    return train_result_df, df_test,


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model on AVA Dataset')
    parser.add_argument('-d', '--dataset-dir', type=str, default=DATASET_DIR, required=False,
                        help='Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=False,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-s', '--sample-size', type=int, default=None, required=False,
                        help='Sample size, None for full size.')
    parser.add_argument('-t', '--model-type', type=str, default=MODEL_BUILD_TYPE[0], required=False,
                        help="Model type to train aesthetic/technical/both.")
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
    arg_model_type = (args.__dict__['model_type']).lower()
    arg_aes_weight_path = args.__dict__['aes_weights_path']
    if arg_aes_weight_path is not None:
        assert os.path.isfile(arg_aes_weight_path), 'Invalid Aesthetic weights, does not exists.'

    arg_tech_weight_path = args.__dict__['aes_weights_path']
    if arg_tech_weight_path is not None:
        assert os.path.isfile(arg_tech_weight_path), 'Invalid Technical weights, does not exists.'

    arg_freeze_base = args.__dict__['freeze_base']
    arg_freeze_base = True if arg_freeze_base.lower() == 'true' else False
    arg_batch_size = args.__dict__['batch_size']
    arg_sample_size = args.__dict__['sample_size']
    arg_epochs = args.__dict__['epochs']
    arg_verbose = args.__dict__['verbose']
    arg_metrics = args.__dict__['metrics']

    # Train the aesthetic model
    if arg_model_type in [MODEL_BUILD_TYPE[0], 'both']:
        aes_train_result_df, aes_test_df, aes_weight_file = train_aesthetic_model(p_model_name=arg_model_name,
                                                                                  p_dataset_dir=arg_dataset_dir,
                                                                                  p_sample_size=arg_sample_size,
                                                                                  p_weight_path=arg_aes_weight_path,
                                                                                  # p_freeze_base_model=arg_freeze_base,
                                                                                  p_batch_size=arg_batch_size,
                                                                                  p_metrics=['accuracy'],
                                                                                  p_epochs=arg_epochs,
                                                                                  p_verbose=arg_verbose)
    # Train the technical model
    if arg_model_type in [MODEL_BUILD_TYPE[1], 'both']:
        tech_train_result_df, tech_test_df, tech_weight_file = train_technical_model(p_model_name=arg_model_name,
                                                                                     p_dataset_dir=arg_dataset_dir,
                                                                                     p_sample_size=arg_sample_size,
                                                                                     p_weight_path=arg_aes_weight_path,
                                                                                     # p_freeze_base=arg_freeze_base,
                                                                                     p_batch_size=arg_batch_size,
                                                                                     # p_metrics=['mean_absolute_error'],
                                                                                     p_epochs=arg_epochs,
                                                                                     p_verbose=arg_verbose)
