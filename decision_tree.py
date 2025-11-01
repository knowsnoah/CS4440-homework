# -------------------------------------------------------------------------
# AUTHOR: Noah Ojeda
# FILENAME: decision_tree.py
# SPECIFICATION: This program trains decision tree classifiers using three cheat-training datasets,
# tests each model on cheat_test.csv, and reports average accuracy.
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: ~2hr 
# -----------------------------------------------------------*/
#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM
#importing some Python libraries

from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']
for ds in dataSets:
    X = []
    Y = []
    accuracies = []
    df = pd.read_csv(ds, sep=',', header=0) #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5Darray X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. 
    #The feature Marital Status must be one-hot-encoded and Taxable Income must be converted to a float.
    for row in data_training:
        #refund value -> 0 or 1
        refund_value = 1 if row[0] == "Yes" else 0

        #marital status 
        single = 1 if row[1] == "Single" else 0
        married = 1 if row[1] == "Married" else 0
        divorced = 1 if row[1] == "Divorced" else 0

        #Taxable income converted into a float
        income_str = str(row[2])
        income_float = float(income_str.replace('k',''))

        #appending these values into X
        X.append([refund_value, single, married, divorced, income_float])

    #transform the original training classes to numbers and add them to the vectorY. 
    # For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    for row in data_training:
        class_label = row[3]
        Y.append(1 if class_label == "Yes" else 0)

    #loop your training and test tasks 10 times here
    for i in range (3):
        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)
        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Married',
        'Divorced', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        #read the test data and add this data to data_test NumPy
        #--> add your Python code here
        # data_test =
        df_test = pd.read_csv('cheat_test.csv', sep=',', header=0) 
        data_test = np.array(df_test.values)[:,1:] #dropping the id column

        #variables to keep track of correct and total predictions
        correct = 0
        total = 0

        for data in data_test:
            #transform the features of the test instances to numbers following the same strategy done during training, 
            # and then use the decision tree to make the class prediction.
            # For instance: class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as 
            # the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            refund_value_test = 1 if data[0] == "Yes" else 0

            single_test = 1 if data[1] == "Single" else 0
            married_test = 1 if data[1] == "Married" else 0
            divorced_test = 1 if data[1] == "Divorced" else 0

            income_str_t = str(data[2])
            income_float_test = float(income_str_t.replace('k',''))

            class_predicted = clf.predict([[refund_value_test, single_test, married_test, divorced_test, income_float_test]])[0]

            #compare the prediction with the true label (located at data[3]) of the test instance to start 
            # calculating the model accuracy.
            #--> add your Python code here
            true_value = 1 if data[3] == "Yes" else 0

            if class_predicted == true_value:
                correct+=1
            total+=1

        #find the average accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        accuracy = correct / total if total else 0.0 #if total doesnt exist or is 0
        accuracies.append(accuracy)
        print(f"run {i+1} accuracy on {ds}: {accuracy:.3f}")
    
    #final accuracy after the 10 runs for each training set
    print(f"final accuracy when training on {ds}: {sum(accuracies)/len(accuracies):.3f}")
