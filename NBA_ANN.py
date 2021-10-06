# -*- coding: utf-8 -*-

# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Seasons_Stats.csv')

# Data Manipulation
df["MPG"] = round(df["MP"] / df["G"], 1) # Minutes Per Game
df["PPG"] = round(df["PTS"] / df["G"], 1) # Points Per Game
df["RPG"] = round(df["TRB"] / df["G"], 1) # Rebound Per Game
df["DRPG"] = round(df["DRB"] / df["G"], 1) # Defensive Rebound Per Game
df["ORPG"] = round(df["ORB"] / df["G"], 1) # Offensive Rebound Per Game
df["APG"] = round(df["AST"] / df["G"], 1) # Assists Per Game
df["SPG"] = round(df["STL"] / df["G"], 1) # Steals Per Game
df["BPG"] = round(df["BLK"] / df["G"], 1) # Blocks Per Game
df["TPG"] = round(df["TOV"] / df["G"], 1) # Turnovers Per Game
df["PFPG"] = round(df["PF"] / df["G"], 1) # Personal Fouls Per Game
df = df[(df["Year"] >= 1980)] # Before 1980 we had a lot of missing values in our data
df = df[(df["MPG"] >= 38)] # Increasing the minutes per game can maybe help to improve the accuracy
df = df[["Pos", "MPG", "PPG", "RPG", "DRPG", "ORPG", "APG", "SPG", "BPG", "TPG", "PFPG"]]
X = df.iloc[:, :11].values
y = df.iloc[:, :1].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ec = df.iloc[:, :1]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ec = ct.fit_transform(ec)
X = np.array(ct.fit_transform(X), dtype=np.float)
y = X[:, :5]
X = X[:, 5:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
new_prediction = classifier.predict(sc.transform(np.array([[39, 31, 4, 3.6, 0.4, 5, 2.5, 1, 3, 1.5]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
