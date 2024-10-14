import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\Debansu\All documents in Data science\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Social_Network_Ads.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifire = LogisticRegression()
classifire.fit(x_train,y_train)

y_pred = classifire.predict(x_test)

from sklearn.metrics import confusion_matrix
metrics = confusion_matrix(y_test, y_pred)
print(metrics)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


from sklearn.metrics import classification_report
Classification = classification_report(y_test, y_pred)
print(Classification)

bias = classifire.score(x_train, y_train)
print(bias)

#----------------------------Future prediction--------------------------------

dataset1 = pd.read_csv(r"D:\Debansu\All documents in Data science\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Future prediction1.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
m = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2['y_pred1'] = classifire.predict(m)

d2.to_csv('pred_Purchased.csv')

import os
os.getcwdb()

