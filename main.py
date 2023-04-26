#Lets just do our imports that we will be using for this code
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_absolute_error, confusion_matrix

# parameters/variables needed to do the kernal and Kfold
kernel = 'linear'
cost = .5
g = 100
x0 = 1
k = 10
Accuracy = 0
MAE = 0
Accuracy2 = 0
MAE2 = 0


#loading the data set given to us
dataset = pd.read_csv('data.csv',sep=";")
dataset.head()
dataset.describe()

#we are doing to do x and y values
dataset.drop('STUDENT ID', 1, inplace=True)
x = dataset.drop('GRADE', 1)
y = dataset['GRADE']

#adding the KFold import
from sklearn.model_selection import KFold
kfold = KFold(n_splits=k)

# kfold.get_n_splits(x) <---- this prints out how many, which is 10

classifier = svm.SVC(kernel=kernel, C=cost, gamma=g, degree=3, coef0=x0)
classifier2 = svm.SVC(kernel='linear', C=cost)

for trainSet, testSet in kfold.split(x):
    train_x, test_x = x.iloc[trainSet], x.iloc[testSet]
    train_y, test_y = y.iloc[trainSet], y.iloc[testSet]

    model = classifier.fit(train_x, train_y)
    pred = model.predict(test_x)
    Accuracy += sum(confusion_matrix(test_y, pred).diagonal())/len(pred)
    MAE += mean_absolute_error(test_y, pred)

    model = classifier2.fit(train_x, train_y)
    pred = model.predict(test_x)
    Accuracy2 += sum(confusion_matrix(test_y, pred).diagonal())/len(pred)
    MAE2 += mean_absolute_error(test_y, pred)
print(Accuracy/k)
print(f"{MAE/k}\n")

print(Accuracy2/k)
print(f"{MAE2/k}")


