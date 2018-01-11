import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("/Users/amit.sood/Documents/Analytics/kaggle/ChurnAnalysis/");
train = pd.read_csv('../ChurnAnalysis/xgbDataTrain.csv')
test = pd.read_csv('../ChurnAnalysis/xgbDataTest.csv')
X = train.iloc[:, 2:25].values
y = train.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = Sequential()
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#rmsprop
classifier.fit(X_train, y_train, batch_size = 30, epochs = 5)



TestSet = test.iloc[:, 2:25].values
TestSetTransformed = sc.transform(TestSet)
y_pred1 = classifier.predict(TestSetTransformed)
test['is_churn'] = y_pred1
test[['msno','is_churn']].to_csv('submissionfinalfire1.csv', index=False)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 50, epochs = 200)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train, cv=10, n_jobs=-1)



#Grid search
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
    
parameters ={'batch_size' :[25, 32] , 'epochs' :[50, 100]}

grid_search = GridSearchCV(estimator= classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train,y_train)