#-------------------------------------------------------------------------
# AUTHOR: Noah Ojeda
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4440- Assignment #4
# TIME SPENT: 
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv 
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#funciton to discretize the real values to the closest class
def discretize(value):
    closest_class = min(classes, key=lambda x: abs(x - value))
    return closest_class

#reading the training data
#--> add your Python code here
filename_training = 'weather_training.csv'
df = pd.read_csv(filename_training, sep=',', header=0) #elimating the header

X_training = df.iloc[:, :-1].values

Y_training_real = df.iloc[:, -1].values

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
Y_training = np.array([discretize(value) for value in Y_training_real])

#reading the test data
#--> add your Python code here
filename_test = 'weather_test.csv'
df_test = pd.read.csv(filename_test, sep=',', header=0) #elimating the header again

X_test = df_test.iloc[:, :-1].values

Y_test_real = df_test.iloc[:, -1].values
#update the test class values according to the discretization (11 values only)
#--> add your Python code here
Y_test = np.array([discretize(value) for value in Y_test_real])

#loop over the hyperparameter value (s)
#--> add your Python code here

for :

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=?)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here



