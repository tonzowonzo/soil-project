# Import libraries
import pandas as pd
import numpy as np

# Columns visible with vision algorithm.
columns = ["mottles", "structure", "texture", "primary_colour", "strength", "secondary_colour", "plasticity"]

# Read in the csv.
df = pd.read_csv(r"C:/Users/Tim/Desktop/SOILAIPROJECT/soil_plasticity_data_encoded.csv")
df = df[columns]

# Show the value counts of plasticity.
print(df["plasticity"].value_counts())
print(df["primary_colour"].value_counts())
# Split the X and y data.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split to train and test data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train an algorithm to test accuracy.
from sklearn.ensemble import RandomForestClassifier
rand_for = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the classifier.
rand_for.fit(X_train, y_train)

# Predict y_test.
y_pred = rand_for.predict(X_test)

# What's the accuracy?
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Try with xgboost.
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=500, objective="multi:softmax", max_depth=3)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# Try with nearest neighbours.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)



