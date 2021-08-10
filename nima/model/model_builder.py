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

from nima.config import MODELS_JSON_FILE_PATH, WEIGHTS_DIR, MODEL_BUILD_TYPE, print_msg, INPUT_SHAPE, CROP_SHAPE
from nima.model.loss import earth_movers_distance


def get_models_dict():
    """
    Get the Models dictionary from the JSON file
    :return:
    """
    import json
    with open(MODELS_JSON_FILE_PATH) as model_json_file:
        models = json.load(model_json_file)
    return models


def model_weight_name(model_name, model_type=MODEL_BUILD_TYPE[0], freeze_base_model=True):
    """
    Returns the name of the model based on the given parameters.
    Use this method to make sure naming convention is followed
    :param model_name: model name, should be in models.json
    :param model_type: NIMA model type
    :param freeze_base_model: Freebase model
    :return: model weight file name
    """
    models = get_models_dict()
    assert model_name in models.keys(), f"Invalid model name {model_name}, should have one of the value {models.keys()}"
    base_model_name = models[model_name]['model_class']
    weight_filename = f'{base_model_name}_{model_type}' \
                      f'{"_all-freezed" if freeze_base_model else "_all-trained"}'
    return weight_filename


class NIMA:

    def __init__(self, base_model_name, base_cnn_weight='imagenet', weights=None, freeze_base_model=True,
                 model_type=MODEL_BUILD_TYPE[0], input_shape=(224, 224, 3), loss=earth_movers_distance, metrics=None):
        """
        Constructor method
        :rtype: NIMA class object - A deep Learning CNN Model
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        self.base_cnn_weight = base_cnn_weight
        assert model_type in MODEL_BUILD_TYPE, f"Invalid model type, should be from {MODEL_BUILD_TYPE}"
        self.model_type = model_type
        if metrics is None:
            metrics = ['accuracy']
        self.metrics = metrics
        self.learning_rate = 0.001
        self.weights = weights
        self.freeze_base_model = freeze_base_model
        self.input_shape = input_shape
        self.loss = loss
        self.model_name = base_model_name
        self.model_class_name = None
        self.base_model = None
        self.model = None
        # Set the model properties.
        self._get_base_module()

    def _get_base_module(self):
        """
        Get the base model based on the base model name
        :return: Base models' library
        """
        # import json
        # with open(MODELS_JSON_FILE_PATH) as model_json_file:
        #     models = json.load(model_json_file)
        models = get_models_dict()
        assert self.model_name in models.keys(), f"Invalid model name {self.model_name}, should have one of the value {models.keys()}"
        self.model_class_name = models[self.model_name]['model_class']
        model_package = models[self.model_name]['model_package']
        self.base_module = importlib.import_module(model_package)
        print_msg(f"NIMA Base CNN module - {model_package}.{self.model_class_name}", 1)

    def freeze_all_layers(self):
        print_msg('Freezing all base CNN layers.', 1)
        for layer in self.base_model.layers:
            layer.trainable = False

    def train_all_layers(self):
        print_msg('Allowing training on base CNN layers.', 1)
        for layer in self.base_model.layers:
            layer.trainable = True

    def build(self):
        """
        Build the CNN model for Neural Image Assessment
        """
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.model_class_name)
        # Set the model properties
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.base_cnn_weight,
                                   pooling='avg', include_top=False)

        # add dropout and dense layer
        x = Dropout(.2)(self.base_model.output)

        # check the model type if aesthetic or technical
        if self.model_type == 'aesthetic':
            x = Dense(10, activation='softmax')(x)
        else:
            x = Dense(1, activation='relu')(x)

        # Assign the class model
        self.model = Model(self.base_model.input, x)

    def compile(self):
        """ Compile the Model """
        self.model.compile(optimizer=Adam(self.learning_rate), loss=self.loss, metrics=self.metrics)
        # Load existing weight and Compile the Model
        if self.weights is not None:
            print_msg(f'Loading existing weights file {self.weights}', 1)
            # self.model.compile(optimizer=Adam(self.learning_rate), loss=self.loss, metrics=self.metrics)
            self.model.load_weights(weights=self.weights)
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
        fig.suptitle(f'{self.model_type.capitalize()} - {self.model_class_name}', fontsize=10,
                     weight='bold', color='green')
        fig.set_figheight(6)
        fig.set_figwidth(10)


    def get_naming_prefix(self):
        weight_filename = f'{self.model_class_name}_{self.model_type}' \
                          f'{"_all-freezed" if self.freeze_base_model else "_all-trained"}'
        return weight_filename

    def _get_liveloss_plot(self, liveloss_filename):
        print_msg(f'Figure path : {liveloss_filename}', 1)
        plot = MatplotlibPlot(figpath=liveloss_filename, before_plots=self._liveloss_before_subplot)
        return plot

    def get_callbacks(self, verbose):
        es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=verbose)
        lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=verbose)
        model_save_name = self.get_naming_prefix()
        # Live loss
        liveloss_filename = os.path.join(WEIGHTS_DIR, model_save_name + ".png")
        plot_loss = PlotLossesKerasTF(outputs=[self._get_liveloss_plot(liveloss_filename)])
        # Model checkpoint
        weight_filepath = os.path.join(WEIGHTS_DIR, model_save_name + ".hdf5")
        checkpoint = ModelCheckpoint(
            filepath=weight_filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
        )

        print_msg(f'Model Weight path : {weight_filepath}', 1)
        # csv logger
        log_filepath = os.path.join(WEIGHTS_DIR,model_save_name + "_log.csv")
        csv_log = CSVLogger(log_filepath)
        print_msg(f'Model log path : {log_filepath}', 1)
        return [es, checkpoint, lr, csv_log, plot_loss]

    def train_model(self, train_generator, validation_generator,
                    epochs=32, verbose=0):
        """
        Train the final model for the given number of parameters
        :param train_generator: Training Image Data Generator
        :param validation_generator: Validation Image Data Generator
        :param epochs: Number of epochs
        :param verbose: verbose
        :return: Training result_df and saved models weight path
        """
        callbacks = self.get_callbacks(verbose)
        # start training
        start_time = time.perf_counter()
        history = self.model.fit(train_generator, validation_data=validation_generator,
                                 epochs=epochs, callbacks=callbacks,
                                 verbose=verbose, use_multiprocessing=True)
        end_time = time.perf_counter()
        print_msg(f'Training Time (HH:MM:SS) : {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}', 1)

        result_df = pd.DataFrame(history.history)
        return result_df

    def evaluate_model(self, df_test, test_generator):
        predictions = self.model.predict(test_generator)
        df_test['y_pred'] = predictions
        loss, acc = self.model.evaluate()
        return df_test, loss, acc
