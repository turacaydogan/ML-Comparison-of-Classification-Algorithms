!wget -O Wholesale customers data.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
%matplotlib inline
df=pd.read_csv("Wholesale")
x=df[['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].values
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
y=df[['Channel']].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

#decision tree
from sklearn.tree import DecisionTreeClassifier
model_tree=DecisionTreeClassifier(criterion='entropy', max_depth=4)
model_tree.fit(x_train,y_train)
pred=model_tree.predict(x_test)

from sklearn import metrics
import matplotlib.pyplot as plt
print("jaccard: ", jaccard_score(y_test, pred))
print("Accuracy Score: ", metrics.accuracy_score(y_test, pred))

#KNN
from sklearn.neighbors import KNeighborsClassifier
k=6
neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train.ravel())
pred=neigh.predict(x_test)
from sklearn import metrics
print('train acc:', metrics.accuracy_score(y_train, neigh.predict(x_train)))
print('test acc:', metrics.accuracy_score(y_test,pred))
print("jaccard: ", jaccard_score(y_test, pred))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR=LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train.ravel())
pred = LR.predict(x_test)
pred_prob=LR.predict_proba(x_test)
print("jaccard", jaccard_score(y_test, pred))

#SVM
from sklearn import svm
model=svm.SVC(kernel='rbf')
model.fit(x_train,y_train.ravel())
pred=model.predict(x_test)
print('jaccard',jaccard_score(y_test, pred))
