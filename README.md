Human-Activity-Monitoring-and-Prediction
========================================
The software package contains two python files:
data_preprocess.py-The script used to transform the data set. The original dataset in stored in four files: X_train.txt, y_train.txt, X_test.txt and y_test.txt. Random Forest in Mahout need the data to be stored in .csv files, and the label value “y” should be attached to the end of each line after all the X variables. This script reads data from the original dataset and transforms it into Mahout-readable dataset.
analysis.py-The script used to perform SVM, KNN and Random Forest by python. To run the script, attach “X_train.txt”, “y_train.txt”, “X_test.txt” and “y_test.txt” under the same directory. The output is the parameters of the best-fit classifiers as well as the test accuracy.

Telecom group
