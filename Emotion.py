# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:01:17 2020

@author: Admin
"""


import pandas as pd
import numpy as np 
import pickle 


df_train = pd.read_csv("C:/Users/Admin/Desktop/Emotion Dataset/train.txt", delimiter = ";", header= None,
                       names=['Sentence','Label'])

df_test = pd.read_csv("C:/Users/Admin/Desktop/Emotion Dataset/test.txt", delimiter = ";", header= None,
                       names=['Sentence','Label'])

df_val = pd.read_csv("C:/Users/Admin/Desktop/Emotion Dataset/val.txt", delimiter = ";", header= None,
                       names=['Sentence','Label'])


df_train.head()

df = pd.concat([df_train, df_test], axis = 0).reset_index(drop = True)
df.head()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Label_enc'] = labelencoder.fit_transform(df['Label'])

df[['Label','Label_enc']].drop_duplicates(keep = 'first')


# Independent variable:
X = df['Sentence']

## Get the Dependent features
y=df['Label_enc']
y.head()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

messages=df.copy()
messages.head()

messages['Sentence'][1]

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Sentence'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[1]

## Also check for lemmitizaton for improve accuracy:(pe.stem)

## Applying Countvectorizer
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

messages.columns

y=messages['Label_enc']

## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=0)

cv.get_feature_names()[:20]
cv.get_params()


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

from sklearn import metrics
import numpy as np
import itertools


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cm, classes=['0','1','2','3','4','5'])

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score

y_train.shape


### Multinnommial classifier with hyperparamaeter:
classifier=MultinomialNB(alpha=0.1)

previous_score=0

for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))

## High Score:
classifier = MultinomialNB(alpha=0.6)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


## Predictions:
def predict_sentiment(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = cv.transform([final_review]).toarray()
  return classifier.predict(temp)


# Predicting values
sample_review = 'i started to see a concerning pattern i d rush home at the end of the evening s activities to write out a post sometimes i d be feeling frustrated and flustered while sometimes i was eager and inspired'

if predict_sentiment(sample_review)== 4:
  print('This is a SAD review.')
elif predict_sentiment(sample_review) == 0:
  print('This is a ANGER review.')
elif predict_sentiment(sample_review) == 3:
  print('This is a LOVE review.')
elif predict_sentiment(sample_review) == 5:
  print('This is a SURPRISE review.')
elif predict_sentiment(sample_review) == 1:
  print('This is a FEAR review.')  
else:
  print('This is a JOY review!')



## saving model to disk:
pickle.dump(classifier, open('Emotion.pkl','wb'))

## Loading model to compare the result:
model = pickle.load(open('Emotion.pkl','rb'))











