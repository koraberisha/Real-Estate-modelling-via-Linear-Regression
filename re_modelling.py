import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


train_dataset = pd.read_csv('real-state/train_full_Real-estate.csv')
test_dataset = pd.read_csv('real-state/test_full_Real-estate.csv')

X_train = train_dataset.iloc[:, 1:6].values
y_train = train_dataset.iloc[:, 7].values
train_labels = train_dataset.iloc[:, 0].values


X_test = test_dataset.iloc[:, 1:6].values
y_test = test_dataset.iloc[:, 7].values



regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print('Mean Absolute Error:' +  str(metrics.mean_absolute_error(y_test, y_pred)) + ' %')
print('Mean Squared Error:'+  str(metrics.mean_squared_error(y_test, y_pred)) + ' %')
print('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + ' %')
y_train_binary = []
for n1 in y_train:
    if n1 >= 30:
        y_train_binary.append(1)
    else:
        y_train_binary.append(0)

y_test_binary = []
for n2 in y_test:
    if n2 >= 30:
        y_test_binary.append(1)
    else:
        y_test_binary.append(0)
        


gnb = GaussianNB()
model = gnb.fit(X_train, y_train_binary)
preds = gnb.predict(X_test)

print("Accuracy of Naive-Bayes classifier: " + str(100*metrics.accuracy_score(y_test_binary, preds)) + " %")
