from src.model.model_builder import NIMA

model = NIMA(base_model_name='mobilenet', loss='categorical_crossentropy')
model.build()
