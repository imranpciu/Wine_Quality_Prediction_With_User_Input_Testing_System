#import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv 
from matplotlib import pyplot 
import matplotlib.pyplot as plt
from pandas import set_option 
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler, LabelEncoder


filename ='winequality-red.csv' 
data = pd.read_csv(filename) 
print(data.head(20))
print(data.shape)
data.info()
set_option('display.width', 100) 
set_option('precision', 3) 
#print (data.describe()) 
#print(data.groupby('quality').size())
#print( data.corr(method='pearson') )

data.hist()
scatter_matrix(data)
pyplot.show()


bins = (2, 6, 8)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'])


data['quality'].value_counts()
sns.countplot(data['quality'])


pyplot.show()
array = data.values 
X = array[:,0:11] 
Y = array[:,11]
validation_size = 0.20 
seed=10
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
models = [] 
models.append(('SVM', SVC())) 
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('RFC', RandomForestClassifier())) 


results = [] 
names = [] 
for name, model in models: 
    kfold = KFold(n_splits=10, random_state=seed) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') 
    results.append(cv_results) 
    names.append(name) 
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
fig = pyplot.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
pyplot.boxplot(results) 
ax.set_xticklabels(names) 
pyplot.show()


rfc = RandomForestClassifier() 
rfc.fit(X_train, Y_train) 
predictions = rfc.predict(X_validation) 
print(accuracy_score(Y_validation, predictions)) 
print(confusion_matrix(Y_validation, predictions)) 
print(classification_report(Y_validation, predictions))


#User input System:








