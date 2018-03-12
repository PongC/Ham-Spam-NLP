# -*- coding: utf-8 -*-

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

#separate ham/spam array
ham=[]
spam=[]
for i in range(len(dataset)):
    if(dataset['a'][i]=='ham'):
        dataset['a'][i]=0
        ham.append(dataset['strings'][i])
    else:
        dataset['a'][i]=1
        spam.append(dataset['strings'][i])
ham_df = pd.DataFrame(ham,columns=['strings'])
spam_df = pd.DataFrame(spam,columns=['strings'])

#clean the dataframes into corpus
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ham_corpus = []
for i in range(0, len(ham_df)):
    strings = re.sub('[^a-zA-Z]', ' ', ham_df['strings'][i])
    strings = strings.lower()
    strings = strings.split()
    ps = PorterStemmer()
    strings = [ps.stem(word) for word in strings if not word in set(stopwords.words('english'))]
    strings = ' '.join(strings)
    ham_corpus.append(strings)
    
spam_corpus = []
for i in range(0, len(spam_df)):
    strings = re.sub('[^a-zA-Z]', ' ', spam_df['strings'][i])
    strings = strings.lower()
    strings = strings.split()
    ps = PorterStemmer()
    strings = [ps.stem(word) for word in strings if not word in set(stopwords.words('english'))]
    strings = ' '.join(strings)
    spam_corpus.append(strings)    

#ham vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20, analyzer='word')
cv_addr = cv.fit_transform(ham_corpus)
ham_vector = pd.SparseDataFrame(cv_addr, columns=cv.get_feature_names(), default_fill_value=0)

tmp=[]
for col in cv.get_feature_names():
    tmp.append([col,sum(ham_vector[col])])
ham_gdf = pd.DataFrame(tmp,columns=['word','frequency']).sort_values(by=['frequency'],
                       ascending=False).reset_index()
del ham_gdf['index']

#spam vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20, analyzer='word')
cv_addr = cv.fit_transform(spam_corpus)
spam_vector = pd.SparseDataFrame(cv_addr, columns=cv.get_feature_names(), default_fill_value=0)

tmp=[]
for col in cv.get_feature_names():
    tmp.append([col,sum(spam_vector[col])])
spam_gdf = pd.DataFrame(tmp,columns=['word','frequency']).sort_values(by=['frequency'],
                       ascending=False).reset_index()
del spam_gdf['index']

#ham plot
print("ham")
ax = plt.axes()
sns.barplot(x="word", y="frequency", data=ham_gdf)
ax.set_title('HAM')
plt.show()
#spam plot
ax = plt.axes()
print("spam")
sns.barplot(x="word", y="frequency", data=spam_gdf)
ax.set_title('SPAM')
plt.show()

#######################################
##              PLOT
#######################################
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 20, analyzer='word')
#cv_addr = cv.fit_transform(corpus)
#x_2 = pd.SparseDataFrame(cv_addr, columns=cv.get_feature_names(), default_fill_value=0)
#
#ax2=[]
#for col in cv.get_feature_names():
#    ax2.append([col,sum(x_2[col])])
#graph_df = pd.DataFrame(ax2,columns=['word','frequency']).sort_values(by=['frequency'],
#                       ascending=False).reset_index()
#del graph_df['index']
########################################
#
#plt.figure()
#graph_df.plot(kind='bar')
#plt.title('Word frequency')
#plt.xlabel('Words')
#plt.ylabel('Frequency')
##pyplot.savefig(output_filename)
#
####################2####################
#ax2=[]
#for col in cv.get_feature_names():
#    ax2.append(sum(x_2[col]))
#graph_df2 = pd.DataFrame([ax2],columns=cv.get_feature_names())
##plt.figure()
#graph_df2.plot(kind='bar')
#plt.title('Word frequency')
#plt.xlabel('Words')
#plt.ylabel('Frequency')
########################################
#
#sns.barplot(x="word", y="frequency", data=graph_df)

