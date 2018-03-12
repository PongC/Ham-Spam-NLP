# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
colnames=['a','b','c','d','e']
dataset = pd.read_csv('spam.csv',names=colnames, encoding = "ISO-8859-1")
dataset = dataset[1:] #remove the first row
dataset = dataset.fillna('') #change NaN to null string ''
dataset['strings']=dataset['b'].map(str)+dataset['c']+dataset['d']+dataset['e']
dataset = dataset.reset_index()
del dataset['index']
del dataset['b']
del dataset['c']
del dataset['d']
del dataset['e']

for i in range(len(dataset)):
    if(dataset['a'][i]=='ham'):
        dataset['a'][i]=0
    else:
        dataset['a'][i]=1
        
#dataset = dataset[:1000] #cut dataset to only 1000 elements
        
# rearrange the dataset format
#dataset = dataset[list('bcde')].astype(str).sum(1) #combine unname column bcd with a
#dataset.to_frame(name='ABC')
#dataset.columns = ['results','reviews']        
    
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    strings = re.sub('[^a-zA-Z]', ' ', dataset['strings'][i])
    strings = strings.lower()
    strings = strings.split()
    ps = PorterStemmer()
    strings = [ps.stem(word) for word in strings if not word in set(stopwords.words('english'))]
    strings = ' '.join(strings)
    corpus.append(strings)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000)
#cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values
y=y.astype('int')

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
one = 0
zero = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
for yoyo in y_test:
    if(yoyo == 0):
        zero=zero+1
    else:
        one=one+1

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#[TN,FP]
#[FN,TP]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0][0]+cm[1][1])/len(X_test)
print("acc = "+str(accuracy))
precision = cm[1][1]/(cm[1][1]+cm[0][1])
recall = cm[1][1]/(cm[1][1]+cm[1][0])
f1 = 2*precision*recall/(precision+recall)
print("f1 = "+str(f1))
