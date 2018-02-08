# A version where more than just the final layer of the model is changed.

from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

import os
os.chdir(r'C:/Users/Tim/pythonscripts')
# Train the model on new data for a few epochs.
from keras.preprocessing.image import ImageDataGenerator

base_model = Xception(weights='imagenet', include_top=False)

# Create the generators for datasets.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'soilimages/train',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'soilimages/test',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Load in the old pretrained model.
from keras.models import load_model
model = load_model('soilNetPretrained11class3.h5')

# Test how many layers there are in our model
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# Freeze some of the layers
for layer in model.layers[:120]:
    layer.trainable = False

# Allow these layers to train
for layer in model.layers[120:]:
    layer.trainable = True
    
# Recompile the model, use adam with a low LR.
from keras.optimizers import adam
model.compile(optimizer=adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit_generator(training_set, steps_per_epoch=25, epochs=10, validation_data=test_set,
                    validation_steps=25)
# Save the model
model.save('SoilEnhancedPretrained2.h5')

model = load_model('SoilEnhancedPretrained2.h5')
scores = model.evaluate_generator(test_set, steps=10)