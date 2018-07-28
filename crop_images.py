# Crop the images to a particular size to be passed to CNN.
# Import libraries.
import numpy as np
from PIL import Image
import os

def crop_image(img, crop_shape=8):
    image = Image.open(img).convert("RGB")
    image = image.crop((0, 0, crop_shape, crop_shape))
    return image
    
crop_image(r"C:/Users/Tim/pythonscripts/soil_or_nonsoil/train/nonsoil/1.png")

def crop_and_save_images(path):
    for file in os.listdir(path):
        cropped_img = crop_image(os.path.join(path, file))
        cropped_img.save(os.path.join(path, "cropped_" + file))
        print(file)
        
crop_and_save_images(r"C:/Users/Tim/pythonscripts/soil_or_nonsoil/test/nonsoil/")
