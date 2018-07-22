# A CNN for 3 class soil texture classification.

# Soil classification with pretrained algorithm.
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import metrics
from keras import losses
import keras
from keras import backend as K
import os
os.chdir(r'C:/Users/Tim/pythonscripts')

# Import the pretrained model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer.
x = Dense(128, activation='relu')(x)

# Add a classifying layer, 4 classes (Softmax classification)
predictions = Dense(4, activation='softmax')(x)

# The model we'll train.
model = Model(inputs=base_model.input, outputs=predictions)

# Train only the top layer, freeze the weights of the others.
for layer in base_model.layers:
    layer.trainable = False
    
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=[metrics.hinge, 'accuracy'])

# Train the model on new data for a few epochs.
from keras.preprocessing.image import ImageDataGenerator

# Create the generators for datasets.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

# Get the image from the directories.
training_set = train_datagen.flow_from_directory(r'soil_logging_colour/train',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'soil_logging_colour/test',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Create callbacks for the model.
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, 
                                         write_graph=True, write_images=True)


# Fit the model.
model.fit_generator(training_set, steps_per_epoch=10, epochs=5, 
                    validation_data=test_set, validation_steps=5,
                    callbacks=[tbCallBack])

# Save the model
model.save('4_colour_soil_model.h5')

# Load the model
from keras.models import load_model
model = load_model('4_colour_soil_model.h5')

# Get the values from the generator
X_test = list(test_set.next())



# Get by class accuracy.
from sklearn.metrics import classification_report
import numpy as np

# Predict from a batch
y_pred = model.predict((X_test[0]))
y_pred = np.argmax(y_pred, axis=1)

# Get y_test vals
y_test = X_test[1]
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred))

