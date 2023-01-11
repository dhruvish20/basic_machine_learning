import numpy as np
from sklearn import preprocessing, svm
import pandas as pd
from sklearn.model_selection import train_test_split

##dataset from kaggle

df=pd.read_csv('breast-cancer.csv')


df.drop(['id'],1,inplace=True)
print(df.head())

X = np.array(df.drop(['diagnosis'],1))
y =  np.array(df['diagnosis'])

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

clf =svm.SVC()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)


             
        
