import importlib
import os
import time

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from livelossplot import PlotLossesKerasTF
from livelossplot.outputs import MatplotlibPlot
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from nima.config import MODELS_JSON_FILE_PATH, WEIGHTS_DIR, BUILD_TYPE_LIST, print_msg
from nima.model.loss import earth_movers_distance


class NIMA:
    def __init__(self, base_model_name, base_cnn_weight='imagenet', weights=None, model_type=BUILD_TYPE_LIST[0],
                 input_shape=(224, 224, 3), loss=earth_movers_distance, metrics=None):
        """
        Constructor method
        :rtype: NIMA class object - A deep Learning CNN Model
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        self.base_cnn_weight = base_cnn_weight
        assert model_type in BUILD_TYPE_LIST, f"Invalid model type, should be from {BUILD_TYPE_LIST}"
        self.model_type = model_type
        if metrics is None:
            metrics = ['accuracy']
        self.learning_rate = 0.001
        self.weights = weights
        self.input_shape = input_shape
        self.model_name = base_model_name
        self.loss = loss
        self.metrics = metrics
        self.base_model_name = None
        self.base_model = None
        self.model = None
        # Set the model properties.
        self._get_base_module(base_model_name)

    def _get_base_module(self, model_name):
        """
        Get the base model based on the base model name
        :param model_name: Base model name
        :return: Base models' library
        """
        import json
        with open(MODELS_JSON_FILE_PATH) as model_json_file:
            models = json.load(model_json_file)
        assert model_name in models.keys(), f"Invalid model name, should have one of the value {models.keys()}"
        self.base_model_name = models[model_name]['model_class']
        model_package = models[model_name]['model_package']
        self.base_module = importlib.import_module(model_package)
        print_msg(f"NIMA Base CNN module - {model_package}.{self.base_model_name}", 1)

    def build(self, freeze_base_model=True):
        """
        Build the CNN model for Neural Image Assessment
        """
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.base_model_name)
        # Set the model properties
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.base_cnn_weight,
                                   pooling='avg', include_top=False)
        # Freeze/UnFreeze base model layers if true
        self.freeze_base_model = freeze_base_model
        if freeze_base_model:
            print_msg('Freezing base CNN layers.', 1)
        else:
            print_msg('Allowing training on base CNN layers.', 1)

        for layer in self.base_model.layers:
            layer.trainable = not freeze_base_model

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
            self.model.compile(optimizer=Adam(self.learning_rate), loss=self.loss, metrics=self.metrics)
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

    def _get_weight_filename(self):
        weight_filename = f'{self.base_model_name}_{self.model_type}' \
                          f'{"_all-freezed" if self.freeze_base_model else "_all-trained"}'
        return weight_filename

    def get_weight_path(self):
        """
        Forms the model's weight path  based on the model name and type
        :rtype: Model's weight path
        """
        weight_filename = f'{self._get_weight_filename()}.hdf5'
        return os.path.join(WEIGHTS_DIR, weight_filename)

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
        # set model weight and path
        weight_filepath = self.get_weight_path()
        train_loss_filename = os.path.join(WEIGHTS_DIR, f'{self._get_weight_filename()}.png')

        print_msg(f'Model Weight path : {weight_filepath}', 1)
        print_msg(f'Figure path : {train_loss_filename}', 1)

        es = EarlyStopping(monitor='val_loss', patience=4, verbose=verbose)
        ckpt = ModelCheckpoint(
            filepath=weight_filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
        )
        lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=verbose)
        plot_loss = PlotLossesKerasTF(outputs=[MatplotlibPlot(figpath=train_loss_filename)])

        # start training
        start_time = time.perf_counter()
        history = self.model.fit(train_generator, validation_data=validation_generator,
                                 epochs=epochs, callbacks=[es, ckpt, lr, plot_loss],
                                 verbose=verbose, use_multiprocessing=True)
        end_time = time.perf_counter()
        print_msg(f'Training Time (HH:MM:SS) : {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}', 1)

        result_df = pd.DataFrame(history.history)
        return result_df, weight_filepath
