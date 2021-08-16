import importlib
import os
import time

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from livelossplot import PlotLossesKerasTF
from livelossplot.outputs import MatplotlibPlot
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from nima.config import MODELS_JSON_FILE_PATH, WEIGHTS_DIR, MODEL_BUILD_TYPE, print_msg, INPUT_SHAPE, CROP_SHAPE, \
    RESULTS_DIR
from nima.model.loss import earth_movers_distance
from nima.utils.preprocess import get_mean_quality_score, normalize_ratings, get_std_score
from nima.utils.tid_dataset_utils import TID_MAX_MEAN_SCORE


def get_models_dict():
    """
    Get the Models dictionary from the JSON file
    :return:
    """
    import json
    with open(MODELS_JSON_FILE_PATH) as model_json_file:
        models = json.load(model_json_file)
    return models


def get_naming_prefix(model_type, model_class_name, prefix=None, freeze=False):
    weight_filename = f'{model_class_name}_{model_type}'
    if prefix is not None:
        weight_filename = weight_filename + '_' + prefix
    if freeze:
        weight_filename = weight_filename + '_freezed'
    else:
        weight_filename = weight_filename + '_all_trained'

    return weight_filename


def get_model_weight_name(model_name, model_type=MODEL_BUILD_TYPE[0], prefix=''):
    """
    Returns the name of the model based on the given parameters.
    Use this method to make sure naming convention is followed
    :param model_name: model name, should be in models.json
    :param model_type: NIMA model type
    :return: model weight file name
    """
    models = get_models_dict()
    assert model_name in models.keys(), f"Invalid model name {model_name}, should have one of the value {models.keys()}"
    model_class_name = models[model_name]['model_class']
    weight_filename = get_naming_prefix(model_type, model_class_name, prefix)
    return weight_filename


def _get_base_module(model_name):
    """ Get the base model based on the base model name :return: Base models' library """
    models = get_models_dict()
    assert model_name in models.keys(), f"Invalid model name {model_name}, " \
                                        f"should have one of the value {models.keys()}"
    model_class_name = models[model_name]['model_class']
    model_package = models[model_name]['model_package']
    base_module = importlib.import_module(model_package)
    print_msg(f"Base CNN module - {model_package}.{model_class_name}", 1)
    return model_class_name, model_package, base_module


def get_callbacks(weights_dir, model_type, model_class_name, prefix=None, liveloss_before_subplot=None,
                  freeze=False, verbose=0, lr_patience=5, es_patience=5, lr_factor=0.95):
    es = EarlyStopping(monitor='val_loss', patience=lr_patience, mode='auto', verbose=verbose)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=es_patience, verbose=verbose)
    model_save_name = get_naming_prefix(model_type, model_class_name, prefix=prefix, freeze=freeze)
    # Live loss
    liveloss_filename = os.path.join(weights_dir, model_save_name + ".png")
    print_msg(f'Figure path : {liveloss_filename}', 1)
    plot = MatplotlibPlot(figpath=liveloss_filename, before_plots=liveloss_before_subplot)
    plot_loss = PlotLossesKerasTF(outputs=[plot])
    # Model checkpoint
    weight_filepath = os.path.join(weights_dir, model_save_name + ".hdf5")
    checkpoint = ModelCheckpoint(
        filepath=weight_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="auto",
        save_best_only=True,
    )

    print_msg(f'Model Weight path : {weight_filepath}', 1)
    # csv logger
    log_filepath = os.path.join(weights_dir, model_save_name + "_log.csv")
    csv_log = CSVLogger(log_filepath)
    print_msg(f'Model log path : {log_filepath}', 1)
    return [es, checkpoint, lr, csv_log, plot_loss]


class NIMA:
    def __init__(self, base_model_name,
                 base_cnn_weight='imagenet', weights_dir=WEIGHTS_DIR,
                 model_lr=3e-7, input_shape=INPUT_SHAPE, crop_size=CROP_SHAPE, tpu_strategy=None):
        """
        Constructor method
        :rtype: NIMA class object - A deep Learning CNN Model
        :param base_model_name: Base model name
        """
        self.tpu_strategy = tpu_strategy
        self.base_module = None
        self.model_name = base_model_name
        self.model_type = MODEL_BUILD_TYPE[0]
        self.model_class_name = None
        self.base_model = None
        self.model = None
        self.weights_dir = weights_dir
        self.loss = earth_movers_distance
        self.metrics = None
        self.model_lr = model_lr
        self.base_cnn_weight = base_cnn_weight
        self.input_shape = input_shape
        self.crop_size = crop_size
        # Set the model properties.
        print_msg(self.model_type)
        self.model_class_name, self.model_name, self.base_module = _get_base_module(self.model_name)
        self.freeze_base_cnn = True

    def freeze_base_layers(self):
        print_msg("Freezing base CNN's layers.", 1)
        for layer in self.base_model.layers:
            layer.trainable = False

    def train_all_layers(self):
        print_msg('Allowing training on base CNN layers.', 1)
        for layer in self.base_model.layers:
            layer.trainable = True

    def _create_model(self, dropout):
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.model_class_name)
        # Set the model properties
        base_model = base_cnn(input_shape=self.input_shape, weights=self.base_cnn_weight,
                              pooling='avg', include_top=False)
        # add dropout and dense layer
        x = Dropout(dropout)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        # Assign the class model
        model = Model(base_model.input, x)
        return base_model, model

    def build(self, dropout=0.75):
        """
        Build the CNN model for Neural Image Assessment
        """
        if self.tpu_strategy is not None:
            with self.tpu_strategy.scope():
                self.base_model, self.model = self._create_model(dropout)
        else:
            self.base_model, self.model = self._create_model(dropout)

    def compile(self):
        """ Compile the Model """
        if self.tpu_strategy is not None:
            with self.tpu_strategy.scope():
                self.model.compile(optimizer=Adam(self.model_lr), loss=self.loss, metrics=self.metrics)
                print_msg("Model compiled successfully with TPU scope .", 1)
        else:
            self.model.compile(optimizer=Adam(self.model_lr), loss=self.loss, metrics=self.metrics)
            print_msg("Model compiled successfully.", 1)

    def get_preprocess_function(self):
        """ Return the model's preprocess_input """
        return getattr(self.base_module, 'preprocess_input')

    def preprocess_input(self, x):
        preprocess_input = self.get_preprocess_function()
        return preprocess_input(x)

    def _liveloss_before_subplot(self, fig: plt.Figure):
        """Add figure title"""
        fig.suptitle(f'{self.model_type} - {self.model_class_name}', fontsize=10,
                     weight='bold', color='green')
        fig.set_figheight(16)
        fig.set_figwidth(12)

    def train_model(self, train_generator, validation_generator,
                    lr_patience=5, es_patience=5, lr_factor=0.95,
                    prefix=None, epochs=32, verbose=0):
        """
        Train the final model for given parameters.
        :param train_generator: Training Data Generator
        :param validation_generator: Validation Data Generator
        :param epochs: Epochs to train
        :param verbose: Verbose mode
        :param lr_factor: ReduceLROnPlateau factor
        :param es_patience: Patience for EarlyStopping
        :param lr_patience: Patience for ReduceLROnPlateau
        :param prefix: prefix string if any
        """
        callbacks = get_callbacks(self.weights_dir, self.model_type, self.model_class_name,
                                  prefix=prefix, verbose=verbose,
                                  lr_patience=lr_patience, es_patience=es_patience, lr_factor=lr_factor,
                                  liveloss_before_subplot=None)
        # start training
        start_time = time.perf_counter()
        history = self.model.fit(train_generator, validation_data=validation_generator,
                                 epochs=epochs, callbacks=callbacks,
                                 verbose=verbose, )
        end_time = time.perf_counter()
        print_msg(f'Training Time (HH:MM:SS) : {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}', 1)

        result_df = pd.DataFrame(history.history)
        return result_df

    def evaluate_model(self, df_test, test_generator, prefix=None, results_dir=RESULTS_DIR):
        # evaluate the model
        eval_result = self.model.evaluate(test_generator)
        print_msg(f"loss({self.loss}) : {eval_result}", 1)

        # predict the values from model, calculate the mean score
        predictions = self.model.predict(test_generator)
        pred_columns = [f'pred_{column}' for column in test_generator.y_col]
        df_pred = pd.DataFrame(predictions, columns=pred_columns)
        df_pred['pred_max_rating'] = df_pred[pred_columns].apply(lambda x: np.argmax(x.to_numpy()) + 1, axis=1)
        df_pred['pred_mean_score'] = df_pred[pred_columns].apply(lambda x:
                                                                 get_mean_quality_score(normalize_ratings(x)), axis=1)
        df_pred['pred_mean_std'] = df_pred[pred_columns].apply(lambda x: get_std_score(normalize_ratings(x)), axis=1)

        df_test = pd.concat([df_test, df_pred], axis=1)

        predict_df_filename = get_naming_prefix(self.model_type,
                                                self.model_class_name
                                                , prefix=prefix) + f'_pred.csv'
        predict_file = os.path.join(results_dir, predict_df_filename)
        print_msg(f'saving predictions to {predict_file}', 1)
        df_test.to_csv(predict_file, index=False)
        return eval_result, df_test


class TechnicalModel:
    def __init__(self, model_name, base_cnn_weight='imagenet', input_shape=INPUT_SHAPE, crop_size=CROP_SHAPE,
                 model_lr=3e-7, weights_dir=WEIGHTS_DIR):
        self.model_name = model_name
        self.model_type = MODEL_BUILD_TYPE[1]
        self.model_class_name = None
        self.base_module = None
        self.base_model = None
        self.model = None
        self.base_cnn_weight = base_cnn_weight
        self.weights_dir = weights_dir
        self.model_lr = model_lr
        self.input_shape = input_shape
        self.crop_size = crop_size
        self.loss = 'mean_squared_error'
        self.metrics = ['mean_absolute_error', 'mean_squared_error']
        self.model_class_name, self.model_name, self.base_module = _get_base_module(self.model_name)
        self.freeze_base_cnn = False

    def build(self, dropout=0.2):
        base_cnn = getattr(self.base_module, self.model_class_name)
        # Set the model properties
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.base_cnn_weight,
                                   pooling='avg', include_top=False)
        x = Dropout(dropout)(self.base_model.output)
        # x = Dense(128, activation='relu')(x)
        x = Dense(1, activation='linear')(x)

        self.model = Model(self.base_model.input, x)

    def freeze_base_layers(self):
        print_msg("Freezing base CNN's layers.", 1)
        for layer in self.base_model.layers:
            layer.trainable = False
        self.freeze_base_cnn = True

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=Adam(learning_rate=self.model_lr), metrics=self.metrics)
        print_msg("Model compiled successfully.", 1)

    def get_preprocess_function(self):
        """
        Return the model's preprocess_input
        """
        return getattr(self.base_module, 'preprocess_input')

    def preprocess_input(self, x):
        preprocess_input = self.get_preprocess_function()
        return preprocess_input(x)

    def _liveloss_before_subplot(self, fig: plt.Figure, axs: np.ndarray, num_lg: int):
        """Add figure title"""
        fig.suptitle(f'{self.model_type} - {self.model_class_name}', fontsize=14,
                     weight='bold', color='green')
        fig.set_figheight(6)
        fig.set_figwidth(10)

    def train_model(self, train_generator, validation_generator,
                    lr_patience=5, es_patience=5, lr_factor=0.95,
                    prefix=None, epochs=32, verbose=0):
        """
        Train the final model for given parameters.
        :param train_generator: Training Data Generator
        :param validation_generator: Validation Data Generator
        :param epochs: Epochs to train
        :param verbose: Verbose mode
        :param lr_factor: ReduceLROnPlateau factor
        :param es_patience: Patience for EarlyStopping
        :param lr_patience: Patience for ReduceLROnPlateau
        :param prefix: prefix string if any
        """
        callbacks = get_callbacks(self.weights_dir, self.model_type, self.model_class_name,
                                  prefix=prefix, verbose=verbose, freeze=self.freeze_base_cnn,
                                  lr_patience=lr_patience, es_patience=es_patience, lr_factor=lr_factor,
                                  liveloss_before_subplot=None)
        # start training
        start_time = time.perf_counter()
        history = self.model.fit(train_generator, validation_data=validation_generator,
                                 epochs=epochs, callbacks=callbacks,
                                 verbose=verbose)
        end_time = time.perf_counter()
        print_msg(f'Training Time (HH:MM:SS) : {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}', 1)

        result_df = pd.DataFrame(history.history)
        return result_df

    def evaluate_model(self, df_test, test_generator, prefix=None, results_dir=RESULTS_DIR):
        # evaluate model
        eval_result = self.model.evaluate(test_generator)
        print_msg(f"loss({self.loss}) : {eval_result[0]} | accuracy({self.metrics}) : {eval_result[1:]}", 1)
        # predict the values from model
        predictions = self.model.predict(test_generator)
        df_test['pred_mean'] = predictions * TID_MAX_MEAN_SCORE
        predict_df_filename = get_naming_prefix(self.model_type,
                                                self.model_class_name,
                                                prefix) + f'_pred.csv'
        predict_file = os.path.join(results_dir, predict_df_filename)
        print_msg(f'saving predictions to {predict_file}', 1)
        df_test.to_csv(predict_file, index=False)
        return eval_result, df_test
