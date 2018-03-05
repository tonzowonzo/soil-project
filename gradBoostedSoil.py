# Import libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

root_dir = 'C:\\Users\\Tim\\pythonscripts\\soilimages'
soil = pd.DataFrame(columns=['soil_image_array', 'soil_type'], index=[x for x in range(1000)])
X = pd.DataFrame(columns=[x for x in range(40000)], index=[x for x in range(1000)])
# Iterate through folders
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        img = cv2.imread(os.path.join(subdir, file), 0)
        soil.soil_image_array[i] = img.tolist()
        soil_name = subdir.split('\\')[-1]
        soil.soil_type[i] = soil_name
        i += 1
        
# Drop NaN values from df
soil = soil.dropna()
   
# Shuffle the dataframe
from sklearn.utils import shuffle
soil = shuffle(soil)

# Encode the y variables
from sklearn.preprocessing import LabelEncoder
y = soil.iloc[:, -1]
le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1, 1)

# Hot encode the labels
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Define both dataframes
soil_array = soil.iloc[:, 0]


#for arry in X:
#    arry.reshape(40000,)
# Reshape values in X
for i, arry in enumerate(X):
    soil_array[i] = np.array(X[i])
    
for i, value in enumerate(X):
    soil_array[i] = value.reshape(40000, 1)
    

#X = np.array(X)

# Get train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train, y_train)


# Tree classifier
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

# X
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
ada_clf.fit(mnist['data'], mnist['target'])
mnist_X = mnist['data']