# A soil CNN from scratch.
'''
Will require approx. 200 images for each soil class, 200 * 12 = 2400 images.
'''
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras import metrics
# Build the model
model = Sequential()  # Instantiate the model.

# Step 1 - Convolution
model.add(Convolution2D(32, (3, 3), activation="relu", input_shape=(299, 299, 3)))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))

# Dropout
model.add(Dropout(0.5))

# Batch norm.
model.add(BatchNormalization())

# Adding a second convolutional layer
model.add(Convolution2D(64, (3, 3), activation="relu"))

# 2nd pooling layer
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))

# Dropout and batch norm.
model.add(Dropout(0.5))
model.add(BatchNormalization())

# Adding a second convolutional layer
model.add(Convolution2D(128, (3, 3), activation="relu"))

# 2nd pooling layer
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))

# Dropout and batch norm.
model.add(Dropout(0.5))
model.add(BatchNormalization())

# Adding a second convolutional layer
model.add(Convolution2D(256, (3, 3), activation="relu"))

# 2nd pooling layer
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))

# Dropout and batch norm.
model.add(Dropout(0.5))
model.add(BatchNormalization())
# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=12, activation="softmax"))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [metrics.hinge])

# Train the model on new data for a few epochs.
from keras.preprocessing.image import ImageDataGenerator

# Create the generators for datasets.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 20,
                                   width_shift_range = 30,
                                   height_shift_range = 30)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'soilimages/train',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'soilimages/test',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set, steps_per_epoch=25, epochs=5, 
                    validation_data=test_set, validation_steps=10)

# Save the model
model.save('soilNetPretrained.h5')

# Load the model
model = load_model('soilNetPretrained.h5')

# Get the values from the generator
X_test = list(test_set.next())

# Predict from a batch

y_pred = model.predict((X_test[0]))


