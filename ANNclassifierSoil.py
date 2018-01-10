# An ANN classifier for soil class.
# Trying to classify soil based on world soil data
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import argmax

# Load in csv dataset
df = pd.read_excel('HWSD_DATA.xlsx')
df = df[df['ISSOIL'].isin([1])] # Drop all the non-soils
df = df[df.SU_CODE90.notnull()] # Drop all non-classified soils.
categorical_vars = ['SU_CODE74', 'SU_CODE85', 'SU_CODE90', 'T_TEXTURE', 'DRAINAGE', 'AWC_CLASS',
                    'ROOTS', 'IL', 'SWR', 'ADD_PROP']
# Get rid of all of the useless columns
df = df.iloc[:, 7:]

# Fill NaN values with medians.
df.fillna(df.median(), inplace=True)

# Add new feature of absolute soil class
df['SOIL_CLASS'] = [x[:2] for x in df['SU_SYM90']]
# Amount of each type of soil

# Get rid of more columns I don't need
df = df.iloc[:, 4:]
'''
Below shows that most our soil types have sample sizes of <20 and are largely spread from
0 to 200, there is one soil type I which almost contains 1000 samples however.
'''
df['SU_SYM90'].unique()
x = df['SU_SYM90'].value_counts()
plt.hist(x, bins=20)

'''
Soil class value counts, similar to above but for absolute classes instead of minimum.
'''
df['SOIL_CLASS'].unique()
_ = df['SOIL_CLASS'].value_counts()
plt.hist(x)
y = df['SOIL_CLASS']
'''
Below shows the distribution of drainage types with 1 being very poor drainage and
6 being somewhat excessive. The most common class is 4 which is a moderately-well drained
soil.
'''
plt.hist(df['DRAINAGE'], bins=6)

'''
Below shows the distribution of the texture types. 2 is the most common texture by
a significant margin, 2 is a medium texture.
'''
plt.hist(df['T_TEXTURE'], bins=4)

'''
Below shows the distributions of each type of soil particulate matter. Gravel is rarely
over 10% in soils while sand commonly makes up over 30% of a soil.
'''
sns.pairplot(data=df, vars=['T_CLAY', 'T_SILT', 'T_SAND', 'T_GRAVEL'])
plt.hist(df['T_CLAY'].dropna(), bins=20)
plt.hist(df['T_SILT'].dropna(), bins=20)
plt.hist(df['T_SAND'].dropna(), bins=20)
plt.hist(df['T_GRAVEL'].dropna(), bins=20)

# Encoding categorical variable for soil class
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1, 1) # It needs reshaping for the onehotencoder to work.
# One hot encode the variable
onehot = OneHotEncoder(sparse=False)
y = onehot.fit_transform(y)

x = df.iloc[:, 2:-1]

# Create x and y train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Turn X_train and y_train to numpy arrays

# Import keras classes
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adadelta
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu', input_shape=(44, )))
classifier.add(Dropout(0.3))

# adding the 2nd hidden layer.
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))

#Adding the output layer
classifier.add(Dense(units=29, kernel_initializer='uniform', activation='softmax'))
   
#Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train.as_matrix(), y_train, batch_size=100, epochs=25)

y_pred = classifier.predict(X_test)
# Prediction metrics.
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print('Recall: {}, Precision: {}, Accuracy: {}'.format(recall, precision, accuracy))
