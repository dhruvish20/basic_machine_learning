import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


##dataset from kaggle

df=pd.read_csv('breast-cancer.csv')

df_corr= df.corr()
df.drop(['id'],1,inplace=True)
print(df.head())

X = np.array(df.drop(['diagnosis'],1))
y =  np.array(df['diagnosis'])

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

clf =neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

plt.figure(figsize=(8, 8))
sns.heatmap(df_corr, cbar=True, annot=False, yticklabels=df.columns,
            cmap=ListedColormap(['#C71585', '#DB7093', '#FF00FF', '#FF69B4', '#FFB6C1', '#FFC0CB']),
            xticklabels=df.columns)
plt.show()
             
        
