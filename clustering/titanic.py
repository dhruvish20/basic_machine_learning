##import matplotlib.pyplot as plt
##from matplotlib import style
##style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df=pd.read_csv('tested.csv')
df.drop(['Name','Ticket'],1,inplace=True)
##df.infer_objects(convert_numeric=True)
df.fillna(0,inplace=True)
##print(df.head())


def handle_nonnumeric_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_nonnumeric_data(df)

df.drop(['Sex'], 1, inplace=True)

X = np.array(df.drop(['Survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
