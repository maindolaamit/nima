import importlib
from keras.models import Model
from keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam

from loss import earth_movers_distance


class NIMA:
    def __init__(self, base_model_name, weights='imagenet',
                 input_shape=(224, 224), loss=earth_movers_distance):
        """
        Constructor method
        :param base_model_name: Base model name
        :param weights: Weights of the model, initialized to imagenet
        """
        models = {'mobilenet': 'MobileNetV2',
                  'nasnet': 'NASNetMobile',
                  'efficientnet': 'EfficientNetB3'}
        if base_model_name not in models:
            raise Exception(f"Invalid model name, should have one of the value {models.keys()}")
        self.base_model_name = models[base_model_name]
        self.weights = weights
        self.input_shape = input_shape
        self.base_module = self.__get_base_module(self.base_model_name)
        self.base_model = None
        self.model = None
        self.loss = loss

    @staticmethod
    def __get_base_module(model_name):
        """
        Get the base model based on the base model name
        :param model_name: Base model name
        :return: Base models' library
        """
        return importlib.import_module(f'tensorflow.keras.applications.{model_name}')

    def build_(self):
        """
        Build the CNN model for Neural Image Assessment
        """
        # Load pre trained model
        base_cnn = getattr(self.base_module, self.base_model_name)
        # Set the model properties
        self.base_model = base_cnn(input_shape=self.input_shape, weights=self.weights,
                                   pooling='avg', include_top=False)

        # add dropout and dense layer
        x = Dropout(.5)(base_cnn.output)
        x = Dense(10, activation='softmax')(x)
        # Assign the class model
        self.model = Model(self.base_model.input, x)

    def compile(self):
        """
        Compile the Model
        """
        self.model.compile(optimizer=Adam, loss=self.loss)

    def preprocess_input(self):
        """
        Return the model's preprocess_input
        """
        # return self.model.preprocess_input
        return getattr(self.base_module, 'preprocess_input')
