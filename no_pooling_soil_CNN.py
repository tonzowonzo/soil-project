# Set up current working directory.
import os
os.chdir("C:/Users/Tim/pythonscripts")

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras import metrics
from keras import losses

# Build the model
model = Sequential()  # Instantiate the model.

# Step 1 - Convolutions
model.add(Convolution2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(Convolution2D(32, (3, 3,), activation="relu"))

# Pool.
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

# More convolutions.
model.add(Convolution2D(64, (3, 3,), activation="relu"))
model.add(Convolution2D(64, (3, 3,), activation="relu"))

# Pool.
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

# More convolutions.
model.add(Convolution2D(128, (3, 3,), activation="relu"))
model.add(Convolution2D(128, (3, 3,), activation="relu"))

# Pool.
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

# Dropout
model.add(Dropout(0.5))

# Batch norm.
model.add(BatchNormalization())

# Dropout
model.add(Dropout(0.5))

# Batch norm.
model.add(BatchNormalization())

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=1024, activation="relu"))

model.add(Dropout(0.5))
model.add(Dense(units=3, activation="softmax"))

# Compiling the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy'
              , metrics = ['accuracy'])

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

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'soil_logging/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'soil_logging/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# Create callbacks for the model.
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_soil_types', histogram_freq=0, 
                                         write_graph=True, write_images=True)

# Fit the model.
model.fit_generator(training_set, steps_per_epoch=10, epochs=10, 
                    validation_data=test_set, validation_steps=10,
                    callbacks=[tbCallBack])

# Save the model
model.save('no_pool_soil_model.h5')

# Load the model
model = load_model('no_pool_soil_model.h5')

# Get the values from the generator
X_test = list(test_set.next())

# Predict from a batch

y_pred2 = model.predict((X_test[0]))