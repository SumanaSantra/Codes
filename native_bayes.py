import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg=pd.read_csv('naivetest.csv',names=['message','label'])
print('The dimensions of the dataset',msg.shape)

print(msg)

msg['labelnum']=msg.label.map({'pos':1,'neg':0})
x=msg.message
y=msg.labelnum

xtrain,xtest,ytrain,ytest=train_test_split(x,y)

print('\nThe total no. of Training Data : ',ytrain.shape)
print('\nThe total no. of Test Data : ',ytest.shape)

cv=CountVectorizer()
xtrain_dtm=cv.fit_transform(xtrain)
xtext_dtm=cv.transfrom(xtest)
print('\nThe words or toxens in the text documents\n')
print(cv.get_feature_names())

df=pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names())

clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)
print('\nAccuracy of the classifier is ',metrics.accuracy_score(ytest,predicted))
print('\nContusim Matric')
print(metrics.confusion_matrix(ytest,predicted))
print('\nPrecision : ',metrics.precision_score(ytest,predicted))
print('\nRecall : ',metrics.recall_score(ytest,predicted))
