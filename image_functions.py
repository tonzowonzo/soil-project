# A set of image functions to use on files.
import os
import cv2
from scipy import ndimage, misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
plt.axis('off')

os.chdir(r'C:/Users/Tim/pythonscripts/soil/') # Enter your current directory here.
path = r'C:/Users/Tim/pythonscripts/soil/' # Enter the path for files you want to manipulate.
files = os.listdir(path) # Create a list of the files present to operate on.

def resizeImage(x_dim, y_dim):
    for i in files:
        img = cv2.imread(i, cv2.COLOR_BGR2GRAY)
        resizedImg = cv2.resize(img, (x_dim, y_dim), interpolation=cv2.INTER_AREA)
        cv2.imwrite(i, resizedImg)

def rotateImage(degrees_to_rotate_by, rotate_from):
    '''
    Where degrees_to_rotate_by is the amount by which the limit is rotated, ie if it is 180
    it will flip the image upsidedown.
    
    rotate_from is the file number that you want to rotate from using the enumerate i.
    ''' 

    for i, imageName in enumerate(files):
        
        if i >= rotate_from:        
            img = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
            cols, rows = img.shape[1], img.shape[0]
            rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), degrees_to_rotate_by, 1)
            rotatedImg = cv2.warpAffine(img, rotationMatrix, (cols, rows))
            cv2.imwrite(str(i + rotate_from) + '.jpg', rotatedImg)
            
def flipImage(rotate_from):
    '''
    Mirror an image.
    
    rotate_from is the file number that you want to rotate from using the enumerate i.
    '''
    for i, imageName in enumerate(files):
        img = Image.open(imageName)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img.save(str(i + rotate_from) + '.jpg')        
        
def addNoiseToImage(add_noise_from=0, noise_type='GAUSSIAN'):
    ''' 
    Adds Gaussian, poisson or speckle noise to an image.
    
    add_noise_from is the file number that you want to rotate from using the enumerate i.
    
    noise_type is the noise method you would like to use:
        gaussian, poisson or speckle
    '''
    for i, imageName in enumerate(files):
        img = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
        row, col, channel = img.shape
        if noise_type.upper() == 'GAUSSIAN':
            mean = (0, 0, 0)
            var = 0.001
            sigma = (var**0.5, var**0.5, var**0.5)
            gauss = np.random.normal(mean, sigma, (row, col, channel))
            gauss = gauss.reshape(row, col, channel)
            noisy = img + gauss
            cv2.imwrite(str(i + add_noise_from) + '.jpg', noisy)
            return noisy, img, gauss

 '''
 It's not actually required to use these functions, you can actually use ImageDataGenerator from the keras library for real time image
 augmentation.
 '''

        

        
