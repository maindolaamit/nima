import argparse
import os

from nima.config import AVA_DATASET_DIR, WEIGHTS_DIR
from nima.model.data_generator import NimaDataGenerator
from nima.model.model_builder import NIMA
from nima.utils.ava_dataset_utils import load_data, get_rating_columns

# PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# WEIGHTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'nima', 'weights')
# AVA_DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')
# AVA_IMAGES_DIR = ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on AVA Dataset')
    parser.add_argument('-d', '--dataset-dir', type=str, default=AVA_DATASET_DIR, required=False,
                        help='AVA Dataset directory.')
    parser.add_argument('-n', '--model-name', type=str, default='mobilenet', required=False,
                        help='Model Name to train, view models.json to know available models for training.')
    parser.add_argument('-s', '--sampe-size', type=int, default=None, required=False,
                        help='Sample size, None for full size.')
    parser.add_argument('-m', '--metrics', type=list, default=['accuracy'], required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-w', '--weights-path', type=str, default=None, required=False,
                        help='Weights file path, if any.')
    parser.add_argument('-b', '--batch-size', type=int, default=64, required=False, help='Batch size.')
    parser.add_argument('-e', '--epochs', type=int, default=15, required=False,
                        help='Number of epochs, default 10.')
    parser.add_argument('-v', '--verbose', type=int, default=0, required=False, help='Verbose, default 0.')
    args = parser.parse_args()

    # Set the AVA Dataset directory, default to current project data directory
    arg_dataset_dir = args.__dict__['dataset_dir']
    assert os.path.isdir(arg_dataset_dir), f'Invalid dataset directory {arg_dataset_dir}'
    ava_images_dir = os.path.join(arg_dataset_dir, 'images')

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
    df_train, df_valid, df_test = load_data(ava_images_dir, arg_sample_size)
    # Form the NIMA Model
    nima_cnn = NIMA(base_model_name=arg_model_name, weights='imagenet', input_shape=(224, 224, 3),
                    metrics=arg_metrics)
    nima_cnn.build()
    # load model weights if existing
    if arg_weight_path is not None:
        nima_cnn.model.load_weights(arg_weight_path)
    
    nima_cnn.compile()
    nima_cnn.summary()

    x_col, y_cols = 'image_id', get_rating_columns()
    # Get the generator
    train_generator = NimaDataGenerator(df_train, ava_images_dir, x_col, y_cols,
                                        preprocess_input=nima_cnn.preprocessing_function(),
                                        is_train=True, batch_size=32, )
    valid_generator = NimaDataGenerator(df_valid, ava_images_dir, x_col, y_cols,
                                        preprocess_input=nima_cnn.preprocessing_function(),
                                        is_train=True, batch_size=32, )

    # # set model weight and path
    # weight_filename = f'{arg_model_name}_weight_best.hdf5'
    # weight_filepath = os.path.join(WEIGHTS_DIR, weight_filename)
    # print(f'Model Weight path : {weight_filepath}')
    #
    # es = EarlyStopping(monitor='val_loss', patience=4, verbose=arg_verbose)
    # ckpt = ModelCheckpoint(
    #     filepath=weight_filepath,
    #     save_weights_only=True,
    #     monitor="val_f2_score",
    #     mode="auto",
    #     save_best_only=True,
    # )
    # lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1)
    # plot_loss = PlotLossesCallback()
    #
    # # start training
    # history = nima.model.fit(train_generator, validation_data=valid_generator,
    #                          epochs=arg_epochs, callbacks=[es, ckpt, lr, plot_loss],
    #                          verbose=arg_verbose)
    # result_df = pd.DataFrame(history.history)
    # Train the model
    result_df, train_weights_file = nima_cnn.train_model(train_generator, valid_generator, WEIGHTS_DIR)

    print(result_df)

    # Form the test NIMA Model
    test_nima_cnn = NIMA(base_model_name=arg_model_name, weights='imagenet', input_shape=(224, 224, 3),
                    metrics=arg_metrics)
    test_nima_cnn.build()
    test_nima_cnn.model.load_weights(train_weights_file)
    test_nima_cnn.compile()

    test_generator =  NimaDataGenerator(df_test, ava_images_dir, x_col, y_cols,
                                        preprocess_input=test_nima_cnn.preprocessing_function(),
                                        is_train=False, batch_size=64, )
    test_nima.model.predict()
