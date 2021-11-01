# -*- coding: utf-8 -*-

# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Seasons_Stats.csv')

"""
PG : 0 0 1 0 0
SG : 0 0 0 0 1
SF : 0 0 0 1 0
PF : 0 1 0 0 0
 C : 1 0 0 0 0
"""

# Data Manipulation
df["MPG"] = round(df["MP"] / df["G"], 1) # Minutes Per Game
df["PPG"] = round(df["PTS"] / df["G"], 1) # Points Per Game
df["DRPG"] = round(df["DRB"] / df["G"], 1) # Defensive Rebounds Per Game
df["ORPG"] = round(df["ORB"] / df["G"], 1) # Offensive Rebounds Per Game
df["APG"] = round(df["AST"] / df["G"], 1) # Assists Per Game
df["SPG"] = round(df["STL"] / df["G"], 1) # Steals Per Game
df["BPG"] = round(df["BLK"] / df["G"], 1) # Blocks Per Game
df["TPG"] = round(df["TOV"] / df["G"], 1) # Turnovers Per Game
df["PFPG"] = round(df["PF"] / df["G"], 1) # Personal Fouls Per Game
df["2P"] = round(df["2P"] / df["G"], 1) # 2-Point Field Goals Per Game
df["2PA"] = round(df["2PA"] / df["G"], 1) # 2-Point Field Goal Attempts Per Game
df["FT"] = round(df["FT"] / df["G"], 1) # Free Throws Per Game
df["FTA"] = round(df["FTA"] / df["G"], 1) # Free Throw Attempts Per Game

# Player must play 70% of his team's games
df1 = df[(df["Year"] >= 1978) & (df["Year"] <= 1998) & (df["G"] >= 58)]
df2 = df[(df["Year"] == 1999) & (df["G"] >= 35)]
df3 = df[(df["Year"] >= 2000) & (df["Year"] <= 2011) & (df["G"] >= 58)]
df4 = df[(df["Year"] == 2012) & (df["G"] >= 47)]
df5 = df[(df["Year"] >= 2013) & (df["G"] >= 58)]

df = pd.concat((df1, df2, df3, df4, df5), axis = 0)
df = df[["Pos", "PER", "TS%", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "OBPM", "DBPM", "VORP","G", "MPG", "PPG", "DRPG", "ORPG", "APG", "SPG", "BPG", "TPG", "PFPG", "2P", "2PA", "eFG%", "FT", "FTA"]]
null = df.isnull()
correlation = df.corr()
x = df.iloc[:, :32].values
y = df.iloc[:, :1].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ec = df.iloc[:, :1]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ec = ct.fit_transform(ec)
x = np.array(ct.fit_transform(x), dtype=np.float)
y = x[:, :5]
x = x[:, 5:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))

# Adding the second hidden layer
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
# new_prediction = classifier.predict(sc.transform(np.array([[]])))
# new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
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
