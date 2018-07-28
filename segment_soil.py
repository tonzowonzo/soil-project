# Segment soil from an image.
# Import libraries.
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load in the model.
model = load_model('soil_or_nonsoil_model.h5')

# Load in an image.
img = Image.open(r"C:/Users/Tim/Desktop/SOILAIPROJECT/Soil_Image/3.jpg").convert("RGB")
img = np.array(img)
plt.imshow(img)

# Image shape.
img_shape = img.shape

# Scale the image for prediction.
img = img / 255

# Create an empty array for predictions.
prediction_array = np.empty(shape=(img_shape[0], img_shape[1]))

# X and y of image.
x = 0
y = 0

for pixel in range(int(img_shape[0]/8 * img_shape[1]/8)):
    
    if pixel % 1000 == 0:
        print(str(pixel * 8) + " pixels have been predicted")
    
    # Update x and y of image when row is predicted.
    if x >= img_shape[0] - 8:
        x = 0
        y += 8
    
    # If at the bottom of the image break the loop.
    elif y >= img_shape[1] - 8:
        break
    
    # Get the 8 by 8 pixel example required for the prediction.
    example = img[x:x+8, y:y+8, :]
    example = np.expand_dims(example, axis=0)
    # Add the most likely class to the prediction array.
    prediction_array[x:x+8, y:y+8] = np.argmax(model.predict(example, batch_size=1))
    x +=8

# Turn prediction array into integers.
prediction_array = prediction_array.astype(int)

# Visualise the results.
plt.figure(figsize=(12, 12))
plt.imshow(prediction_array, cmap="binary")
plt.show()

# Masked image.
masked_img = img.copy()
mask = np.ma.masked_where(prediction_array, prediction_array != 1)
masked_img[mask] = 0
plt.figure(figsize=(12, 12))
plt.imshow(masked_img, cmap="binary")
plt.show()