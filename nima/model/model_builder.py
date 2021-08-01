import importlib
import os
from pathlib import Path

from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from nima.model.loss import earth_movers_distance

MODELS_FILE_DIR = Path(__file__).resolve().parent
MODELS_JSON_FILE_PATH = os.path.join(MODELS_FILE_DIR, 'models.json')


class NIMA:
    def __init__(self, base_model_name, weights='imagenet',
                 input_shape=(224, 224, 3), loss=earth_movers_distance, metrics=None):
        """
        Constructor method
        :rtype: NIMA class object - A deep Learning CNN Model
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        if metrics is None:
            metrics = ['accuracy']
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
        print(f"Model's module - {model_package}.{self.base_model_name}")

    def build(self):
        """
        Build the CNN model for Neural Image Assessment
        """
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.base_model_name)
        # Set the model properties
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.weights,
                                   pooling='avg', include_top=False)
        # add dropout and dense layer
        x = Dropout(.2)(self.base_model.output)
        x = Dense(10, activation='softmax')(x)
        # Assign the class model
        self.model = Model(self.base_model.input, x)

    def compile(self):
        """
        Compile the Model
        """
        for layer in self.model.layers:
            layer.trainable = False
        self.model.compile(optimizer=Adam(), loss=self.loss)

    def preprocessing_function(self):
        """
        Return the model's preprocess_input
        """
        return getattr(self.base_module, 'preprocess_input')
