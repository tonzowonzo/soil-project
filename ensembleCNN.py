# Import 3 CNN's
from keras.models import load_model, Model, Input
from keras.layers import Average
import numpy as np
import pandas as pd

model1 = load_model(insert_model1_here) # Inception v3.
model2 = load_model(insert_model2_here) # Inception resnet v2.
model3 = load_model(insert_model3_here) # VGG19, reshape input tensor to (299, 299, 3) or (3, 299, 299).

# Built the ensemble model
models = [model1, model2, model3]
def ensemble(models, model_input):
  outputs = [model.outputs[0] for model in models] # Gives us the output values so we can take their averages.
  y = Average()(outputs)
  model = Model(model_input, y, name='ensemble')
  return model

ensemble_models = ensemble(models, model_input)

# Evaluate the error.
def evaluate_error(model):
  pred = model.predict(X_test, batch_size=32)
  pred = np.expand_dims(pred, axis=1)
  error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
  return error

evaluate_error(ensemble_models)


