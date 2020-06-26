from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets

iris=datasets.load_iris()
x=iris.data
y=iris.target
print('sepal-length','sepal-width','petal-length','petal-width')
print(x)
print('class:0- Iris-Sentosa ,1-Iris-Versicolor, 2-Iris-Virginica')
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
Classifier=KNeighborsClassifier(n_neighbors=3)
Classifier.fit(x_train,y_train)
y_pred=Classifier.predict(x_test)
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Matrix')
print(classification_report(y_test,y_pred))
