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
df['USDA_SOIL_CLASS'] = np.nan
# Get rid of all of the useless columns
df = df.iloc[:, 7:]

# Fill NaN values with medians.
df.fillna(df.median(), inplace=True)
# Add USDA soil classes.
def USDA_LABEL(row):
    if row in ['FL', 'RG', 'GL']:
        return 'Entisol'
    if row in ['AC']:
        return 'Ultisol'
    if row in ['CM']:
        return 'Inceptisol'
    if row in ['FR']:
        return 'Oxisol'
    if row in ['HS']:
        return 'Histosol'
    if row in ['CL', 'GY']:
        return 'Aridisol'
    if row in ['AL', 'VR']:
        return 'Vertisol'
    if row in ['PZ']:
        return 'Spodosol'
    if row in ['PD', 'SN', 'PL', 'LV', 'LX']:
        return 'Alfisol'
    if row in ['CH', 'GR', 'KS', 'PH', 'SC']:
        return 'Mollisol'
    if row in ['AN']:
        return 'Andisol'
    else:
        return np.NaN


# Add new feature of absolute soil class
df['SOIL_CLASS'] = [x[:2] for x in df['SU_SYM90']]
# Amount of each type of soil

# Add USDA soil class.
df['USDA_SOIL_CLASS'] = [USDA_LABEL(x) for x in df['SOIL_CLASS']]

# Get rid of more columns I don't need
df = df.iloc[:, 6:-1]

# Drop final NaNs
df = df.dropna()
'''
Below shows that most our soil types have sample sizes of <20 and are largely spread from
0 to 200, there is one soil type I which almost contains 1000 samples however.
'''
df['USDA_SOIL_CLASS'].unique()
x = df['USDA_SOIL_CLASS'].value_counts()
plt.hist(x, bins=20)

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

# Encoding categorical variable for soil class
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y = df.iloc[:, -1]
y = le.fit_transform(y)
y = y.reshape(-1, 1) # It needs reshaping for the onehotencoder to work.
# One hot encode the variable
onehot = OneHotEncoder(sparse=False)
y = onehot.fit_transform(y)

x = df.iloc[:, 0:-1]
columns = list(x.columns.values)

x = df[['DRAINAGE', 'T_USDA_TEX_CLASS', 'T_BULK_DENSITY', 'T_PH_H2O']]
# Some feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x) #need to fit and then xform
#Rename columns with original names
#x.columns = columns
# Create x and y train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Turn X_train and y_train to numpy arrays

# Import keras classes
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adadelta
from keras.layers import Dropout
from keras.layers import BatchNormalization
classifier = Sequential()
classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu', input_shape=(4, )))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())
# adding the 2nd hidden layer.
classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

classifier.add(Dense(units=512, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())
#Adding the output layer
classifier.add(Dense(units=11, kernel_initializer='uniform', activation='softmax'))
   
#Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size=25, epochs=50)

y_pred = classifier.predict(X_test)

# Save the model
classifier.save('ANNsoilUSDA.h5')

