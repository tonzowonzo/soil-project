# Import libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

root_dir = 'C:\\Users\\Tim\\pythonscripts\\soilimages'
soil = pd.DataFrame(columns=['soil_image_array', 'soil_type'], index=[x for x in range(1000)])

# For black and white
#X = pd.DataFrame(columns=[x for x in range(40000)], index=[x for x in range(1000)])

# For colour images
X = pd.DataFrame(columns=[x for x in range(120000)], index=[x for x in range(1000)])
# Iterate through folders
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        img = cv2.imread(os.path.join(subdir, file), 1)
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

#for i, arry in enumerate(soil_array):
#    soil_array[i] = np.array(arry)

#for arry in X:
#    arry.reshape(40000,)
# Reshape values in X
#for i, arry in enumerate(soil_array):
#    soil_array[i] = np.array(X[i])
    
#for i, value in enumerate(soil_array):
#    soil_array[i] = value.reshape(40000, 1)

# Append individual values in list to columns, this is extremely inefficient.
# Use for black and white images 
#for i, array in enumerate(soil_array):
#    count = 0
#    for j in array:
#        for k in j:
#            X[count][i] = k
#            count += 1
#    if i % 10 == 0:
#        print(i)

# Use for colour images
for i, array in enumerate(soil_array):
    count = 0
    for j in array:
        for k in j:
            for l in k:
                X[count][i] = l
                count += 1
                if count % 20000 == 0:
                    print(count)
    if i % 10 == 0:
        print(i)

# Drop NaN values in the dataframe.
X = X.dropna()

# Save dataframe
X.to_pickle('soil_image_arrays.pkl')

# Read in X dataframe again
X = pd.read_pickle('soil_image_arrays.pkl')

# Get train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Adaboost classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                             n_estimators=500, learning_rate=1,
                             algorithm='SAMME.R')
ada_clf.fit(X_train, y_train)

# Gradient boosted classifier.


# Tree classifier
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

from xgboost import XGBClassifier
xg_clf = XGBClassifier(n_estimators=500, objective='multi:softmax', num_class=12)
xg_clf.fit(X_train, y_train)
y_pred = xg_clf.predict(X_test)

# Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(n_estimators=500)
extra_clf.fit(X_train, y_train)
y_pred = extra_clf.predict(X_test)

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)

from sklearn.metrics import  accuracy_score
print(accuracy_score(y_test, y_pred))
