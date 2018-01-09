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

# Random forest.
from sklearn.ensemble import RandomForestClassifier
rand_for = RandomForestClassifier()
rand_for.fit(X_train, y_train)

# Try model on the test set.
from sklearn.model_selection import cross_val_predict, cross_val_score
cross_val_score(rand_for, X_train, y_train, cv=3, scoring='accuracy')

# Check confusion matrix to see if the model is just selection most likely classes.
y_pred = rand_for.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# Trying gradient boosting instead


