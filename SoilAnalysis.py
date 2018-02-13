# Artificial ANN on soil data to predict it's properties from banding.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the datasets
train = pd.read_csv(r'C:/Users/Tim/pythonscripts/datasets/training.csv')
test = pd.read_csv(r'C:/Users/Tim/pythonscripts/datasets/sorted_test.csv')
pred_vars = ['BSAN', 'BSAS', 'BSAV', 'CTI', 'ELEV', 'EVI', 'LSTD', 'LSTN',
             'REF1', 'REF2', 'REF3', 'REF7', 'RELI', 'TMAP', 'TMFI', 'Depth',
             'Ca', 'P', 'pH', 'SOC', 'Sand', 'Depth']

# Drop columns that are not in the test set.
train = train[test.columns]

# Encode columns that aren't numbers.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train.Depth = le.fit_transform(train.Depth)
le2 = LabelEncoder()
test.Depth = le2.fit_transform(test.Depth)

# Define X_train and y_train.
X_train = train.iloc[:, 1: train.columns.get_loc('BSAN')]
y_train = train.iloc[:, train.columns.get_loc('BSAN'):]

# Define X_test and y_test.
X_test = test.iloc[:, 1: test.columns.get_loc('BSAN')]
y_test = test.iloc[:, test.columns.get_loc('BSAN'):]

# Import our random forest regressor.
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)

# Perform grid search to find best hyperparameters.
from sklearn.grid_search import GridSearchCV
param_grid = {'n_estimators': [3, 5, 7, 9], 'bootstrap':[True, False]}
grid_search = GridSearchCV(random_forest, param_grid)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


# Predict values.
y_pred = random_forest.predict(X_test)

# Check metrics of accuracy, recall and precision.
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
