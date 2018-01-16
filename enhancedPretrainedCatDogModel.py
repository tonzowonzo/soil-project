# A version where more than just the final layer of the model is changed.
# Cat vs dog recognition with pretrained algorithm.
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# Train the model on new data for a few epochs.
from keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(weights='imagenet', include_top=False)

# Create the generators for datasets.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'Convolutional_Neural_Networks/dataset/test_set',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Load in the old pretrained model.
from keras.models import load_model
model = load_model('catDogPretrained.h5')

# Test how many layers there are in our model
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# Freeze some of the layers
for layer in model.layers[:298]:
    layer.trainable = False

# Allow these layers to train
for layer in model.layers[298:]:
    layer.trainable = True
    
# Recompile the model, use adam with a low LR.
from keras.optimizers import adam
model.compile(optimizer=adam(lr=0.001), loss='binary_crossentropy')

# Fit the model
model.fit_generator(training_set, steps_per_epoch=25, epochs=5, validation_data=test_set,
                    validation_steps=25)
# Save the model
model.save('catDogPretrainedEnhanced.h5')
#
## Load the model
#from keras.models import load_model
#model = load_model('catDogPretrainedEnhanced.h5')
#
## Get the values from the generator
#X_test = list(test_set.next())
#
## Predict from a batch
#y_pred = model.predict(X_test[0])

