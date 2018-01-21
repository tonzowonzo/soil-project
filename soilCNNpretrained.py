# A CNN for 12 class soil classification.

# Soil classification with pretrained algorithm.
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os
os.chdir(r'C:/Users/Tim/pythonscripts')
# Import the pretrained model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer.
x = Dense(1024, activation='relu')(x)

# Add a classifying layer, 2 classes (Binary classification)
predictions = Dense(12, activation='softmax')(x)

# The model we'll train.
model = Model(inputs=base_model.input, outputs=predictions)

# Train only the top layer, freeze the weights of the others.
for layer in base_model.layers:
    layer.trainable = False
    
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

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
from keras.models import load_model
model = load_model('soilNetPretrained.h5')

# Get the values from the generator
X_test = list(test_set.next())

# Predict from a batch

y_pred = model.predict(preprocess_input(X_test[0]))

