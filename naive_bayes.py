#-------------------------------------------------------------------------
# AUTHOR: Noah Ojeda
# FILENAME: naive_bayes.py
# SPECIFICATION: This program trains a Naïve Bayes classifier on the weather_training.csv dataset, 
# using discretized temperature values as class labels. It performs a grid search over different smoothing parameters, 
# evaluates predictions on weather_test.csv using a ±15% tolerance rule,and reports the highest accuracy found along 
# with the corresponding smoothing value.
# FOR: CS 4440- Assignment #4
# TIME SPENT: ~1hr 
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv 
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#funciton to discretize the real values to the closest class vlaue for the Y labels (continous to discrete)
def discretize(value):
    closest_class = min(classes, key=lambda x: abs(x - value))
    return closest_class

#reading the training data
#--> add your Python code here
filename_training = 'weather_training.csv'
df = pd.read_csv(filename_training, sep=',', header=0) #elimating the header

X_training = df.iloc[:, 1:-1].values.astype(float)

Y_training_real = df.iloc[:, -1].values.astype(float)

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
Y_training = np.array([discretize(value) for value in Y_training_real])

#reading the test data
#--> add your Python code here
filename_test = 'weather_test.csv'
df_test = pd.read_csv(filename_test, sep=',', header=0) #elimating the header again

X_test = df_test.iloc[:, 1:-1].values.astype(float)

Y_test_real = df_test.iloc[:, -1].values.astype(float)
#update the test class values according to the discretization (11 values only)
#--> add your Python code here
Y_test = np.array([discretize(value) for value in Y_test_real])

#loop over the hyperparameter value (s)
#--> add your Python code here
highest_accuracy = 0
best_s = 0

for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, Y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    predictions = clf.predict(X_test)

    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here
    correct = 0
    total = len(Y_test_real)

    for predicted_value, real_value in zip(predictions, Y_test_real):

        if real_value == 0: #to avoid division by zero
            if predicted_value == 0:
                correct += 1

        else:
            percent_diff = 100* abs(predicted_value - real_value)/ (real_value)
            if percent_diff <= 15.0: #if the prediction is within the tolerance given 
                correct += 1

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here
    accuracy = correct / total if total > 0.0 else 0.0 #to avoid division by zero again
    
    #return the best accuracy and the corresponding s value
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_s = s
        print(f"Highest Naive Bayes accuracy so far: {highest_accuracy}\nParameters: s= {best_s}")



