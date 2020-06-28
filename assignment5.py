import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


ad=pd.read_csv('adult.csv',sep=',')

print(ad.head())
print(ad.describe())
print(ad.info())
ad=ad.drop(['fnlwgt','marital-status','relationship','race','gender','capital-gain','capital-loss'],axis=1)
print(ad['education'].astype('category').value_counts())

print(ad['hours-per-week'].astype('category').value_counts())
print(ad['workclass'].astype('category').value_counts())

sbn.distplot(ad['hours-per-week'])
plt.show()

ad.loc[ad['income'] == '<=50K', 'income'] = 50
ad.loc[ad['income'] == '>50K', 'income'] = 40
print(ad['income'].astype('category').value_counts())

categorical = ad.select_dtypes(include=['object'])
print(categorical.head())

numerical = ad.select_dtypes(include=['int64', 'float64'])
print(numerical.head())

y = ad.pop('income')
X = numerical
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=100)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = LogisticRegression()
print(model.fit(X_train, y_train))

y_test = model.predict(X_test)
y_train_pred = model.predict(X_train)
print(metrics.accuracy_score(y_test, y_train_pred))

'''con_metrics = metrics.confusion_matrix(y_test, y_train_pred)
print(con_metrics)

sbn.heatmap(con_metrics, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()'''

print(metrics.classification_report(y_test, y_train_pred))
