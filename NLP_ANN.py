# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab
import seaborn as sns

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
    
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#=================================================
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
cv = CountVectorizer(max_features = 2500, analyzer='word')
#cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values
y=y.astype('int')

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = 2500))

# Adding the second hidden layers
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
#[TN,FP]
#[FN,TP]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate accuracy, precision, and f1 results
accuracy = (cm[0][0]+cm[1][1])/len(X_test)
print("acc = "+str(accuracy))
precision = cm[1][1]/(cm[1][1]+cm[0][1])
recall = cm[1][1]/(cm[1][1]+cm[1][0])
f1 = 2*precision*recall/(precision+recall)
print("f1 = "+str(f1))






