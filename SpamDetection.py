import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('SMSSpamCollection',sep='\t',names=['target','message'])
'''print(df.head())

print(df.columns)

print(df.shape)

print(df['target'].value_counts())
print(df['message'].value_counts())


sbn.countplot(x='target', data=df)
plt.show()'''

'''import nltk
nltk.download('wordnet')'''

'''import nltk
nltk.download('stopwords')'''

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

clean_msg_lst = []
msg_len_lst = []

def preprocess(raw_msg, flag):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_msg)
    letters_only = letters_only.lower()
    words = letters_only.split()
    words = [w for w in words if not w in stopwords.words("english")]
    if(flag == 'stem'):
        words = [stemmer.stem(word) for word in words]
    else:
        words = [lemmatizer.lemmatize(word) for word in words]

    clean_msg_lst.append(" ".join(words))
    msg_len_lst.append(len(words))

from tqdm import tqdm, tqdm_notebook
tqdm.pandas()

clean_msg_lst = []
msg_len_lst = []
df['message'].progress_apply(lambda x: preprocess(x, 'stem'))
df['clean_msg_stem'] = clean_msg_lst
df['msg_length_stem'] = msg_len_lst

clean_msg_lst = []
msg_len_lst = []
df['message'].progress_apply(lambda x: preprocess(x, 'lemma'))
df['clean_msg_stem'] = clean_msg_lst
df['msg_length_stem'] = msg_len_lst
#print(df.head())
#print(df.columns)

df['Spam']=df['target'].apply(lambda x: 0 if x=='spam' else 1)
#print(df.head())

from sklearn.model_selection  import train_test_split
train, test = train_test_split(df,test_size=0.2,random_state=42)
train_clean_msg=[]
for message in train['clean_msg_stem']:
    train_clean_msg.append(message)
test_clean_msg=[]
for message in test['clean_msg_stem']:
    test_clean_msg.append(message)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word")
train_features = vectorizer.fit_transform(train_clean_msg)
test_features = vectorizer.transform(test_clean_msg)
#print(vectorizer.vocabulary_)
print("Total unique words:", len(vectorizer.vocabulary_))
print("Type of train_features:", type(train_features))
print("Shape of input data", train_features.shape)

#Logistic Regression
'''from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
Classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200)]
dense_features = train_features.toarray()

dense_test = test_features.toarray()

for classifier in tqdm(Classifiers):
    fit = classifier.fit(dense_features,train['message'])
    pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['message'])
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))'''


#SVM
'''x = df['msg_length_stem','Spam']
y = df['target']


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
classifier = SVC()
print(classifier.fit(X_train, y_train))
#print(df.columns)

y_pred = classifier.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
con_metric = metrics.confusion_matrix(y_test, y_pred)
print(con_metric)
sbn.heatmap(con_metric, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()'''
