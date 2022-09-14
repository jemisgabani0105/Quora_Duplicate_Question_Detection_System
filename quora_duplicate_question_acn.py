# -*- coding: utf-8 -*-
"""Quora_Duplicate_Question_ACN.ipynb


import warnings
warnings.filterwarnings("ignore")
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,precision_score,accuracy_score
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

"""## Mounting Drive"""

from google.colab import drive
drive.mount("/content/drive/")

"""## Exploratory Data Analysis

### Overview of the Dataset:
 Data set was taken from Kaggle datasets


1.   Total samples were present in questions.csv
2.   questions.csv contains 5 columns namely `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`.
3. Number of samples in the data set are 404,351 as seen in the output of the code below.
"""

df = pd.read_csv("drive/My Drive/questions.csv")


print("Number of data points:",df.shape[0])

"""### Observing the samples:


1.   qid1,qid2,id,is_duplicate are numerical valuess and the actual questions question1 and question2 are text data. So, we need some feature engineering to convert them to some vector representation.
2.   `is_duplicate` contains two classes `0` for `not duplicate` (-ve class) and `1` for `duplicate`(+ ve class).
3. From the observations it is clear that it is a binary class classification problem.
4. We need to predict the i_duplicate value.


"""

df.head()

df_backup=df
df.info()

"""### NULL values:


1.   From the above observation it is clear that there are null values in the `question1` and `question2` text data.
2.   So we need to eliminate the null values before preprocessing the data and feature engineering.

### Duplicate counts:
   In the below o/p we can see that there approximately 61.8% of `is_duplicate=0` values and the remaining of the opposite value.
"""

df.groupby("is_duplicate")['id'].count().plot.bar()

qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
unique_qs = len(np.unique(qids))
qs_morethan_onetime = np.sum(qids.value_counts() > 1)
print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))
#print len(np.unique(qids))

print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))

print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 

q_vals=qids.value_counts()

q_vals=q_vals.values

x = ["unique_questions" , "Repeated Questions"]
y =  [unique_qs , qs_morethan_onetime]

plt.figure(figsize=(10, 6))
plt.title ("Plot representing unique and repeated questions  ")
sns.barplot(x = x,y = y)
plt.show()

pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()

print ("Number of duplicate questions",(pair_duplicates.shape[0] - df.shape[0]))

plt.figure(figsize=(20, 10))

plt.hist(qids.value_counts(), bins=160)

plt.yscale('log')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts())))

"""## Data cleaning and preprocessing:

#### Dropping nan containing samples
"""

df = df.dropna(thresh=6)

df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True) 
a = 0 
for i in range(a,a+10):
    print(df.question1[i])
    print(df.question2[i])
    print()

"""#### Storing target value in variable `target`"""

target=df['is_duplicate']
with open('label', 'wb') as fp:
    pickle.dump(target.values, fp)

len(target)

## Getting the stop words from the engish. There is inbuilt feature to extract that

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):

    # specifictext


    phrase = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", phrase)
    phrase = re.sub(r",", " ", phrase)
    phrase = re.sub(r"\.", " ", phrase)
    phrase = re.sub(r"!", " ", phrase)
    phrase = re.sub(r"\(", " ( ", phrase)
    phrase = re.sub(r"\)", " ) ", phrase)
    phrase = re.sub(r"\?", " ", phrase)
    phrase = re.sub(r"\s{2,}", " ", phrase)

    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub("\'ve", " have ", phrase)
    phrase = re.sub(" whats ", " what is ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub("n't", " not ", phrase)
    phrase = re.sub("i'm", "i am", phrase, flags=re.IGNORECASE)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub("e\.g\.", " eg ", phrase, flags=re.IGNORECASE)
    phrase = re.sub("b\.g\.", " bg ", phrase, flags=re.IGNORECASE)
    phrase = re.sub("e-mail", " email ", phrase, flags=re.IGNORECASE)
    phrase = re.sub("(\d+)(kK)", " \g<1>000 ", phrase)
    phrase = re.sub("\(s\)", " ", phrase, flags=re.IGNORECASE)
    phrase = re.sub("[c-fC-F]\:\/", " disk ", phrase)
    phrase = re.sub('(?<=[0-9])\,(?=[0-9])', "", phrase)  ## removing comma between numbers like 15,000...
    phrase = re.sub('\$', " dollar ", phrase)
    phrase = re.sub('\%', " percent ", phrase)
    phrase = re.sub('\&', " and ", phrase)

    phrase = re.sub("(?<=[0-9])rs ", " rs ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(" rs(?=[0-9])", " rs ", phrase, flags=re.IGNORECASE)

    return phrase

"""### Preprocessing questions"""

preprocessed_question1 = []
preprocessed_question2 = []

import time 
start=time.time()

# tqdm is for printing the status bar
try:
  for sentence in (df['question1'].values):
      encoded_string = sentence.encode("ascii", "ignore")
      sentence = encoded_string.decode()
      sentence = re.sub(r"http\S+", "", sentence)
      sentence = BeautifulSoup(sentence, 'lxml').get_text()
      sentence = decontracted(sentence)
      sentence = re.sub("\S*\d\S*", "", sentence).strip()
      sentence = re.sub('[^A-Za-z]+', ' ', sentence)
      #------------------------------------------------------------------------
      #https://stackoverflow.com/questions/32705962/removing-any-single-letter-on-a-string-in-python    #------------------------------------------------------------------------
      sentence =re.sub(r'\b\w\b', ' ', sentence)# To remove single letter words
      sentence=re.sub(r'\s{2,}', ' ', sentence).strip()

      #print(sentence)
      #------------------------------------------------------------------------
      # https://gist.github.com/sebleier/554280
      sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stop_words)
      preprocessed_question1.append(sentence.strip())

except TypeError:
  print(sentence)

print(f'time taken for completion of data cleaning is {time.time()-start}')

import time 
start=time.time()
try:
  preprocessed_question2 = []
  # tqdm is for printing the status bar
  for sentence in (df['question2'].values):
      sentence = re.sub(r"http\S+", "", sentence)
      sentence = BeautifulSoup(sentence, 'lxml').get_text()
      sentence = decontracted(sentence)
      sentence = re.sub("\S*\d\S*", "", sentence).strip()
      sentence = re.sub('[^A-Za-z]+', ' ', sentence)
      #------------------------------------------------------------------------
      #https://stackoverflow.com/questions/32705962/removing-any-single-letter-on-a-string-in-python    #------------------------------------------------------------------------
      sentence =re.sub(r'\b\w\b', ' ', sentence)# To remove single letter words
      sentence=re.sub(r'\s{2,}', ' ', sentence).strip()

      #print(sentence)
      #------------------------------------------------------------------------
      # https://gist.github.com/sebleier/554280
      sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stop_words)
      preprocessed_question2.append(sentence.strip())

except TypeError:
  print(sentence)

print(f'time taken for completion of data cleaning is {time.time()-start}')

"""#### `Dumping and loading from pickel files` """

with open('question1_cleaned', 'wb') as fp:
    pickle.dump(preprocessed_question1, fp)

with open('question2_cleaned', 'wb') as fp:
    pickle.dump(preprocessed_question2, fp)

file=open('question1_cleaned','rb')
preprocessed_question1=pickle.load(file)

file=open('question2_cleaned','rb')
preprocessed_question2=pickle.load(file)

file=open('label','rb')
labels=pickle.load(file)

len(preprocessed_question1)

len(preprocessed_question2)

labels=target.values

len(labels)

labels

"""# Linear Models:

## Logistic regression on `unigrams`.

    In the paper they clearly mentioned that they used
    L2 regularisation and the alpha is set to 0.00001 and 20 iterations
"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

"""#### Unigrams
    In this we are creating word unigrams using  
    tf-idf  vectorizer. we can also use counttf-idf vectorizer for the same
"""

tfv_q1 =TfidfVectorizer(analyzer='word',ngram_range = (1,1))

from copy import deepcopy

tfv_q2=deepcopy(tfv_q1)

question1 = tfv_q1.fit_transform(preprocessed_question1)
question2 = tfv_q2.fit_transform(preprocessed_question2)

labels=target.values
X = scipy.sparse.hstack((question1,question2))
y = labels

X

type(y)

"""#### Splitting the X and y to train valid and test """

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size = 0.20, random_state = 42,shuffle=True)

with open('X_train_unigram', 'wb') as fp:
    pickle.dump(X_train, fp)


with open('X_valid_unigram', 'wb') as fp:
    pickle.dump(X_valid, fp)

with open('X_test_unigram', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_train_unigram', 'wb') as fp:
    pickle.dump(y_train, fp)


with open('y_valid_unigram', 'wb') as fp:
    pickle.dump(y_valid, fp)

with open('y_test_unigram', 'wb') as fp:
    pickle.dump(y_test, fp)

type(X_train)

X_train.shape

X_train[0].shape

"""### Defining the model with the above mentioned parameters"""

clf = SGDClassifier(loss='log',penalty='l2',alpha=0.00001,max_iter=20,random_state=42)
clf.fit(X_train, y_train)

clf.predict(X_train)

clf.score(X_train,y_train)

clf.score(X_valid, y_valid)

y_pred=clf.predict(X_test)

clf.score(X_test, y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""#### Observations:
    As stated in the paper accuracy is 75.55 on the test data for
    alpha=0.00001 and max_iteration=20
    In the paper on unigrams it was given that the accuracy_score on unigrams if 75.4 and we got 75.55 which are similar.
    But are getting higher f1-score compared to the model i.e in the paper it was given for unigrams the f1-score is 63.8 but we are getting around 74.33

## Logistic Regression on `Bigrams`

### Bigrams
    In this we are creating word bigrams  using  
    tf-idf  vectorizer. we can also use counttf-idf vectorizer for the same

    Reference : https://hub.packtpub.com/use-tensorflow-and-nlp-to-detect-duplicate-quora-questions-tutorial/
"""

tfv_q1 =TfidfVectorizer(analyzer='word',ngram_range = (1,2))

from copy import deepcopy

tfv_q2=deepcopy(tfv_q1)

question1 = tfv_q1.fit_transform(preprocessed_question1)
question2 = tfv_q2.fit_transform(preprocessed_question2)

labels=target.values
X = scipy.sparse.hstack((question1,question2))
y = labels

X

type(y)

"""#### Splitting the X and y to train valid and test """

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size = 0.20, random_state = 42,shuffle=True)

with open('X_train_bigram', 'wb') as fp:
    pickle.dump(X_train, fp)


with open('X_valid_bigram', 'wb') as fp:
    pickle.dump(X_valid, fp)

with open('X_test_bigram', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_train_bigram', 'wb') as fp:
    pickle.dump(y_train, fp)


with open('y_valid_bigram', 'wb') as fp:
    pickle.dump(y_valid, fp)

with open('y_test_bigram', 'wb') as fp:
    pickle.dump(y_test, fp)

type(X_train)

X_train[0].shape

X_train.shape

"""### Defining the model with the above mentioned parameters"""

clf = SGDClassifier(loss='log',penalty='l2',alpha=0.00001,max_iter=20,random_state=42)
clf.fit(X_train, y_train)

clf.predict(X_train)

clf.score(X_train,y_train)

clf.score(X_valid, y_valid)

y_pred=clf.predict(X_test)

clf.score(X_test, y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""#### Observations:

    For alpha=0.00001 and max_iteration=20
    In the paper on bigrams it was given that the accuracy_score on unigrams if 79.5 but we got 77.9  which is tad lower accuracy

## Logistic Regression on `Trigrams`

### Trigrams
    In this we are creating word Trigrams using  
    tf-idf  vectorizer. we can also use counttf-idf vectorizer for the same
"""

tfv_q1 =TfidfVectorizer(analyzer='word',ngram_range = (1,3))

from copy import deepcopy

tfv_q2=deepcopy(tfv_q1)

question1 = tfv_q1.fit_transform(preprocessed_question1)
question2 = tfv_q2.fit_transform(preprocessed_question2)

labels=target.values
X = scipy.sparse.hstack((question1,question2))
y = labels

X

"""#### Splitting the X and y to train valid and test """

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size = 0.20, random_state = 42,shuffle=True)

type(y)

X_train.shape

X_train[0].shape

with open('X_train_trigram', 'wb') as fp:
    pickle.dump(X_train, fp)


with open('X_valid_trigram', 'wb') as fp:
    pickle.dump(X_valid, fp)

with open('X_test_trigram', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_train_trigram', 'wb') as fp:
    pickle.dump(y_train, fp)


with open('y_valid_trigram', 'wb') as fp:
    pickle.dump(y_valid, fp)

with open('y_test_trigram', 'wb') as fp:
    pickle.dump(y_test, fp)

"""### Defining the model with the above mentioned parameters"""

clf = SGDClassifier(loss='log',penalty='l2',alpha=0.00001,max_iter=20,random_state=42,learning_rate='optimal').fit(X, y)
clf.fit(X_train, y_train)

clf.predict(X_train)

clf.score(X_train,y_train)

clf.score(X_valid, y_valid)

y_pred=clf.predict(X_test)

y_pred

clf.score(X_test, y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""#### Observations:

    For alpha=0.00001 and max_iteration=20
    In the paper on trigrams it was given that the accuracy_score on trigrams if 80.8 but we got 78.7 which is tad lower accuracy and only a little bit
    higher than bigrams. So, trigrams is not as efficient as bigrams

    F1-score = 76.8 which is higher compared to the papers f1-score of 71.8

## Experiments on Trigrams with Logistic regression

### Effect of varying the regularisation parameter on the trigram model
"""

alpha=[0.1,0.01,0.001,0.0001,0.00001,0.000001]

validation_accuracies=[]
F1_scores=[]

for i in alpha:
  clf = SGDClassifier(loss='log',penalty='l2',alpha=i,max_iter=20,random_state=42,learning_rate='optimal').fit(X, y)
  clf.fit(X_train, y_train)
  val_score=clf.score(X_valid, y_valid)
  y_pred=clf.predict(X_valid)
  f_score=f1_score(y_valid, y_pred, average='weighted')
  validation_accuracies.append(val_score*100)
  F1_scores.append(f_score*100)

plt.plot(validation_accuracies, label='Accuracies validation dataset')
plt.plot(F1_scores, label='Validation F1-scores loss')
plt.title('Vaidation data set accuracies and f1-scores')
plt.legend();

from prettytable import PrettyTable
z = PrettyTable()

z.field_names = ["Regularization weight", "Validation Set Accuracy%", "F-Scores on validation dataset"]
z.add_row([alpha[0],validation_accuracies[0] ,F1_scores[0]])
z.add_row([alpha[1],validation_accuracies[1] ,F1_scores[1]])
z.add_row([alpha[2],validation_accuracies[2] ,F1_scores[2]])
z.add_row([alpha[3],validation_accuracies[3] ,F1_scores[3]])
z.add_row([alpha[4],validation_accuracies[4] ,F1_scores[4]])
z.add_row([alpha[5],validation_accuracies[5] ,F1_scores[5]])

print(z)

"""#### Observations:
    clearly from the above table we can see that the 
    Accuracy and F1-score is high for alpha = 0.000001 are high which is higher than the one given in paper

### Effect of varying the number of iterations parameter on the trigram model
"""

iterations=[5,10,15,20,25,30,35,40,45,50]

validation_accuracies=[]
F1_scores=[]

for i in iterations:
  clf = SGDClassifier(loss='log',penalty='l2',alpha=0.000001,max_iter=i,random_state=42,learning_rate='optimal')
  clf.fit(X_train, y_train)
  val_score=clf.score(X_valid, y_valid)
  y_pred=clf.predict(X_valid)
  f_score=f1_score(y_valid, y_pred, average='weighted')
  validation_accuracies.append(val_score*100)
  F1_scores.append(f_score*100)

plt.plot(validation_accuracies, label='Accuracies validation dataset')
plt.plot(F1_scores, label='Validation F1-scores loss')
plt.title('Vaidation data set accuracies and f1-scores')
plt.legend();

from prettytable import PrettyTable
s = PrettyTable()

s.field_names = ["Number of iterations", "Validation Set Accuracy%", "F-Scores on validation dataset"]
s.add_row([iterations[0],validation_accuracies[0] ,F1_scores[0]])
s.add_row([iterations[1],validation_accuracies[1] ,F1_scores[1]])
s.add_row([iterations[2],validation_accuracies[2] ,F1_scores[2]])
s.add_row([iterations[3],validation_accuracies[3] ,F1_scores[3]])
s.add_row([iterations[4],validation_accuracies[4] ,F1_scores[4]])
s.add_row([iterations[5],validation_accuracies[5] ,F1_scores[5]])
s.add_row([iterations[6],validation_accuracies[6] ,F1_scores[6]])
s.add_row([iterations[7],validation_accuracies[7] ,F1_scores[7]])
s.add_row([iterations[8],validation_accuracies[8] ,F1_scores[8]])
s.add_row([iterations[9],validation_accuracies[9] ,F1_scores[9]])

print(s)

plt.plot(validation_accuracies, label='Accuracies validation dataset')
plt.plot(F1_scores, label='Validation F1-scores loss')
plt.title('Vaidation data set accuracies and f1-scores')
plt.legend();

"""#### Observations:
    From the iteration 15 onwards the accuracy remains same and the 
    f1-score also remains same. It depended heavily on the alpha parameter which is 0.000001

## Testing the logistic regression on the `best parameters alpha = 0.000001 and max_iterations=15.`
"""

clf = SGDClassifier(loss='log',penalty='l2',alpha=0.000001,max_iter=i,random_state=42,learning_rate='optimal')
clf.fit(X_train, y_train)

"""#### Validation ACCURACY"""

val_score=clf.score(X_valid, y_valid)
val_score

y_pred=clf.predict(X_valid)

"""#### Validation f1-score"""

f_score=f1_score(y_valid, y_pred, average='weighted')
f_score

"""#### Accuracy score and f1-score"""

y_pred=clf.predict(X_test)

y_pred

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

"""#### Observations:
    Clearly accuracy on the test data has 
    increased to 81.2 percent using best parameters.
    f1-score also increased to 79.33 percent which are bit higher compared to the one given in the paper.

## SVM on Unigrams
"""

Linear_SV_Classifier = SVC(kernel='linear' )  # linear kernel SVC

Linear_SV_Classifier.fit(X=X_train[:150000], y=y_train[:150000])

Linear_SV_Classifier.score(X_train[:150000],y_train[:150000])

Linear_SV_Classifier.score(X_valid, y_valid)

y_pred=Linear_SV_Classifier.predict(X_test)

Linear_SV_Classifier.score(X_test, y_test)

"""## SVM on Bigrams"""

Linear_SV_Classifier2 = SVC(kernel='linear' )

tfv_q1 =TfidfVectorizer(analyzer='word',ngram_range = (1,2))

from copy import deepcopy

tfv_q2=deepcopy(tfv_q1)

question1 = tfv_q1.fit_transform(preprocessed_question1)
question2 = tfv_q2.fit_transform(preprocessed_question2)

labels=target.values
X = scipy.sparse.hstack((question1,question2))
y = labels

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size = 0.20, random_state = 42,shuffle=True)

with open('X_train_bigram', 'wb') as fp:
    pickle.dump(X_train, fp)


with open('X_valid_bigram', 'wb') as fp:
    pickle.dump(X_valid, fp)

with open('X_test_bigram', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_train_bigram', 'wb') as fp:
    pickle.dump(y_train, fp)


with open('y_valid_bigram', 'wb') as fp:
    pickle.dump(y_valid, fp)

with open('y_test_bigram', 'wb') as fp:
    pickle.dump(y_test, fp)

Linear_SV_Classifier2.fit(X=X_train, y=y_train)

Linear_SV_Classifier2.score(X_train,y_train)

Linear_SV_Classifier2.score(X_valid, y_valid)

y_pred=Linear_SV_Classifier2.predict(X_test)

Linear_SV_Classifier3.score(X_test, y_test)

"""## SVM on Trigrams"""

Linear_SV_Classifier3 = SVC(kernel='linear' )

tfv_q1 =TfidfVectorizer(analyzer='word',ngram_range = (1,3))

from copy import deepcopy

tfv_q2=deepcopy(tfv_q1)

question1 = tfv_q1.fit_transform(preprocessed_question1)
question2 = tfv_q2.fit_transform(preprocessed_question2)

labels=target.values
X = scipy.sparse.hstack((question1,question2))
y = labels

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size = 0.20, random_state = 42,shuffle=True)

with open('X_train_trigram', 'wb') as fp:
    pickle.dump(X_train, fp)


with open('X_valid_trigram', 'wb') as fp:
    pickle.dump(X_valid, fp)

with open('X_test_trigram', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_train_trigram', 'wb') as fp:
    pickle.dump(y_train, fp)


with open('y_valid_trigram', 'wb') as fp:
    pickle.dump(y_valid, fp)

with open('y_test_trigram', 'wb') as fp:
    pickle.dump(y_test, fp)

Linear_SV_Classifier3.fit(X=X_train, y=y_train)

Linear_SV_Classifier3.score(X_train,y_train)

Linear_SV_Classifier3.score(X_valid, y_valid)

y_pred=Linear_SV_Classifier3.predict(X_test)

Linear_SV_Classifier3.score(X_test, y_test)

"""#### Observations:
    From the above experiments with svm it is evident that the accuracies are very similar.
    SVM on trigrams produced greater accuracy which is expected.
    Although it produced better accuracy for trigrams it takes more time to train and accuracy is similar to logistic regression.
    So for this data set Logistic regression is best since it takes only seconds to train and produced high accuracy compared to the SVM.

#  Tree based methods 
    As discussed in the paper tree based method on such high dimesnions is not 
    good since it is computationally intractable to train. So, we will train on 
    new features.
"""

avg_accuracies=[]
avg_f1_scores=[]

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

"""## Extracting new features"""

q1=df['question1'].values
q2=df['question2'].values

q1

q2

# https://stackoverflow.com/a/47091490/4084039
import re

def decontractTree(phrase):

    # specifictext


    phrase = re.sub(r",", " ", phrase)
    phrase = re.sub(r"\.", " ", phrase)
    phrase = re.sub(r"!", " ", phrase)
    phrase = re.sub(r"\(", " ( ", phrase)
    phrase = re.sub(r"\)", " ) ", phrase)
    phrase = re.sub(r"\?", " ", phrase)


    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub("\'ve", " have ", phrase)
    phrase = re.sub(" whats ", " what is ", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub("n't", " not ", phrase)
    phrase = re.sub("i'm", "i am", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub("e\.g\.", " eg ", phrase)
    phrase = re.sub("b\.g\.", " bg ", phrase)
    phrase = re.sub("e-mail", " email ", phrase)
    phrase = re.sub("(\d+)(kK)", " \g<1>000 ", phrase)
    phrase = re.sub("\(s\)", " ", phrase)
    phrase = re.sub("[c-fC-F]\:\/", " disk ", phrase)
    phrase = re.sub('(?<=[0-9])\,(?=[0-9])', "", phrase)  ## removing comma between numbers like 15,000...
    phrase = re.sub('\$', " dollar ", phrase)
    phrase = re.sub('\%', " percent ", phrase)
    phrase = re.sub('\&', " and ", phrase)

    phrase = re.sub("(?<=[0-9])rs ", " rs ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(" rs(?=[0-9])", " rs ", phrase, flags=re.IGNORECASE)

    return phrase

tree_question1=[]

import time 
start=time.time()

# tqdm is for printing the status bar
try:
  for sentence in (df['question1'].values):
      encoded_string = sentence.encode("ascii", "ignore")
      sentence = encoded_string.decode()
      sentence = re.sub(r"http\S+", "", sentence)
      sentence = BeautifulSoup(sentence, 'lxml').get_text()
      sentence = decontractTree(sentence)
      #sentence = re.sub("\S*\d\S*", "", sentence).strip()
      #sentence = re.sub('[^A-Za-z]+', ' ', sentence)
      #------------------------------------------------------------------------
      #https://stackoverflow.com/questions/32705962/removing-any-single-letter-on-a-string-in-python    #------------------------------------------------------------------------
      sentence =re.sub(r'\b\w\b', ' ', sentence)# To remove single letter words
      sentence=re.sub(r'\s{2,}', ' ', sentence).strip()

      #print(sentence)
      #------------------------------------------------------------------------
      # https://gist.github.com/sebleier/554280
      #sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stop_words)
      tree_question1.append(sentence.strip())

except TypeError:
  print(sentence)

print(f'time taken for completion of data cleaning is {time.time()-start}')

tree_question2=[]

import time 
start=time.time()

# tqdm is for printing the status bar
try:
  for sentence in (df['question2'].values):
      encoded_string = sentence.encode("ascii", "ignore")
      sentence = encoded_string.decode()
      sentence = re.sub(r"http\S+", "", sentence)
      sentence = BeautifulSoup(sentence, 'lxml').get_text()
      sentence = decontractTree(sentence)
      #sentence = re.sub("\S*\d\S*", "", sentence).strip()
      #sentence = re.sub('[^A-Za-z]+', ' ', sentence)
      #------------------------------------------------------------------------
      #https://stackoverflow.com/questions/32705962/removing-any-single-letter-on-a-string-in-python    #------------------------------------------------------------------------
      sentence =re.sub(r'\b\w\b', ' ', sentence)# To remove single letter words
      sentence=re.sub(r'\s{2,}', ' ', sentence).strip()

      #print(sentence)
      #------------------------------------------------------------------------
      # https://gist.github.com/sebleier/554280
      #sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stop_words)
      tree_question2.append(sentence.strip())

except TypeError:
  print(sentence)

print(f'time taken for completion of data cleaning is {time.time()-start}')

features = [0.0]*21

features

len(tree_question1)

#features = [0.0]*21
Div_factor = 0.0001
STOP_WORDS = stopwords.words('english')

def get_features(q1,q2):
    features = [0.0]*21

    #for i in range(len(preprocessed_question1)):
    q1_tokens =  q1.split()
    q2_tokens =  q2.split()


    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return features

    maxx_length = len(q1_tokens) if len(q1_tokens) >= len(q2_tokens) else len(q2_tokens)

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    ################## L features i.e related to Length ################
    features[0] = len(q1_tokens)
    features[1] = len(q2_tokens)
    features[2] = (abs(len(q1_tokens) - len(q2_tokens)))
    features[3] = (len(q1_tokens) / len(q2_tokens))

    ################## LC number of common Lower cased words  ###############
    lc1 = set([lc for lc in q1_tokens if lc.islower()])
    lc2 = set([lc for lc in q1_tokens if lc.islower()])

    count_lc = len(lc1.intersection(lc2))

    features[4] = count_lc
    features[5] = count_lc//maxx_length ## integer division for better results

    ################## LCXS number of common Lower cased words excluding stopwords ###############

    lcxs1 = set([lc for lc in q1_words if lc.islower()])
    lcxs2 = set([lc for lc in q2_words if lc.islower()])

    count_lcxs = len(lcxs1.intersection(lcxs2))

    features[6] = count_lcxs
    features[7] = count_lcxs//maxx_length ## integer division for better results

    ################## Same Last word  ###############

    features[8] = int(q1_tokens[-1] == q2_tokens[-1]) ## either 0 or 1

    ################## CAP number of common CAPITALISED   ###############

    CAP1 = set([lc for lc in q1_tokens if lc.isupper()])
    CAP2 = set([lc for lc in q1_tokens if lc.isupper()])

    count_cap = len(CAP1.intersection(CAP2))

    features[9] = count_cap
    features[10] = count_cap//maxx_length ## integer division for better results

    ##############  Calculating the prefixes of length 3 to 6 along with their count/maxx_length##############
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    for m,n in zip(q1_tokens,q2_tokens):
        if len(m)>=3 and len(n)>=3:
          if m[:3] == n[:3]:
            count_3+=1
          if m[:4] == n[:4]:
            count_4+=1
          if m[:5] == n[:5]:
            count_5+=1
          if m[:6] == n[:6]:
            count_6+=1

    features[11] = count_3
    features[12] = count_3//maxx_length

    features[13] = count_4
    features[14] = count_4//maxx_length

    features[15] = count_5
    features[16] = count_5//maxx_length

    features[17] = count_6
    features[18] = count_6//maxx_length

    ########################### Some Miscellaneous Features ######################

    digits1 = set([d for d in q1_tokens if d.isdigit()])
    digits2 = set([d for d in q2_tokens if d.isdigit()])

    count_digits = len(digits1.intersection(digits2))

    features[19] = 1 if ('not' in q1_tokens) and ('not' in q2_tokens) else 0
    features[20] = 1 if count_digits > 0 else 0

    return np.asarray(features)

len(tree_question2)

features = np.zeros((len(tree_question1),21))

for i in range(len(tree_question2)):

    features[i]= get_features(tree_question1[i],tree_question2[i])

features.shape

X = features
y= labels

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size = 0.20, random_state = 42,shuffle=True)

type(y)

X_train.shape

X_train[0].shape

"""## Features set 1
    Length of question1 and length of question2, difference in length 
    l1-l2. Ratio of length l1/l2. Total number of features =4.

    For all the subsequent features sets we will use indexes to extract certain features to train the model

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train[:,:4], y_train)   ## fitting only first four length based features

clf.predict(X_valid[:,:4])

clf.score(X_train[:,:4],y_train)

clf.score(X_valid[:,:4], y_valid)

y_pred=clf.predict(X_test[:,:4])

clf.score(X_test[:,:4], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train[:,:4], y_train)

clf.predict(X_valid[:,:4])

clf.score(X_train[:,:4],y_train)

clf.score(X_valid[:,:4], y_valid)

y_pred=clf.predict(X_test[:,:4])

clf.score(X_test[:,:4], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train[:,:4], y_train)

clf.predict(X_valid[:,:4])

clf.score(X_train[:,:4],y_train)

clf.score(X_valid[:,:4], y_valid)

y_pred=clf.predict(X_test[:,:4])

clf.score(X_test[:,:4], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((63.9+63.93+63.93)/3)
avg_f1_scores.append((55.93+56.87+56.87)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 63.91 which is bit higher compared to the paper's which is 63.7
    and f1_score in paper is 30.7 which less compared to our model

## New features set 2

    In this additional features such as lower cased common words are included

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train[:,:6], y_train)   ## fitting only first four length based features

clf.predict(X_valid[:,:6])

clf.score(X_train[:,:6],y_train)

clf.score(X_valid[:,:6], y_valid)

y_pred=clf.predict(X_test[:,:6])

clf.score(X_test[:,:6], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train[:,:6], y_train)

clf.predict(X_valid[:,:6])

clf.score(X_train[:,:6],y_train)

clf.score(X_valid[:,:6], y_valid)

y_pred=clf.predict(X_test[:,:6])

clf.score(X_test[:,:6], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train[:,:6], y_train)

clf.predict(X_valid[:,:6])

clf.score(X_train[:,:6],y_train)

clf.score(X_valid[:,:6], y_valid)

y_pred=clf.predict(X_test[:,:6])

clf.score(X_test[:,:6], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((64.47+64.47+63.31)/3)
avg_f1_scores.append((58.40+58.39+59.02)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 64.08 which is bit low  compared to the paper's which is 68.5
    and f1_scores are almost similar

## New Features set 3
    In this we'll include features such as
    Lower cased common words excluding the stop words

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train[:,:8], y_train)   ## fitting only first four length based features

clf.predict(X_valid[:,:8])

clf.score(X_train[:,:8],y_train)

clf.score(X_valid[:,:8], y_valid)

y_pred=clf.predict(X_test[:,:8])

clf.score(X_test[:,:8], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train[:,:8], y_train)

clf.predict(X_valid[:,:8])

clf.score(X_train[:,:8],y_train)

clf.score(X_valid[:,:8], y_valid)

y_pred=clf.predict(X_test[:,:8])

clf.score(X_test[:,:8], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train[:,:8], y_train)

clf.predict(X_valid[:,:8])

clf.score(X_train[:,:8],y_train)

clf.score(X_valid[:,:8], y_valid)

y_pred=clf.predict(X_test[:,:8])

clf.score(X_test[:,:8], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((67.82+68.41+68.30)/3)
avg_f1_scores.append((67.99+68.38+68.34)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 68.17 which is bit higher compared to the paper's which is 70.7
    and f1_score of 68.23 is pretty high compared to the papers which is 63.3

## New Features set 4
    In this we included the additional features which are Last word being same

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train[:,:9], y_train)   ## fitting only first four length based features

clf.predict(X_valid[:,:9])

clf.score(X_train[:,:9],y_train)

clf.score(X_valid[:,:9], y_valid)

y_pred=clf.predict(X_test[:,:9])

clf.score(X_test[:,:9], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train[:,:9], y_train)

clf.predict(X_valid[:,:9])

clf.score(X_train[:,:9],y_train)

clf.score(X_valid[:,:9], y_valid)

y_pred=clf.predict(X_test[:,:9])

clf.score(X_test[:,:9], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train[:,:9], y_train)

clf.predict(X_valid[:,:9])

clf.score(X_train[:,:9],y_train)

clf.score(X_valid[:,:9], y_valid)

y_pred=clf.predict(X_test[:,:9])

clf.score(X_test[:,:9], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((70.30+70.85+70.84)/3)
avg_f1_scores.append((70.11+70.37+70.37)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 70.66 which is bit lower compared to the paper's which is 72.7
    and f1_scores of 70.28 is higher compared to the paper which is 63.6

## New Features Set 5
    In this additional features such as common capitalised words are included

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train[:,:11], y_train)   ## fitting only first four length based features

clf.predict(X_valid[:,:11])

clf.score(X_train[:,:11],y_train)

clf.score(X_valid[:,:11], y_valid)

y_pred=clf.predict(X_test[:,:11])

clf.score(X_test[:,:11], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train[:,:11], y_train)

clf.predict(X_valid[:,:11])

clf.score(X_train[:,:11],y_train)

clf.score(X_valid[:,:11], y_valid)

y_pred=clf.predict(X_test[:,:11])

clf.score(X_test[:,:11], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train[:,:11], y_train)

clf.predict(X_valid[:,:11])

clf.score(X_train[:,:11],y_train)

clf.score(X_valid[:,:11], y_valid)

y_pred=clf.predict(X_test[:,:11])

clf.score(X_test[:,:11], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((70.37+70.91+70.88)/3)
avg_f1_scores.append((70.19+70.52+70.48)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 70.72 which is bit low compared to the paper's which is 72.8
    and f1_score of 70.39 is higher compared to paper's which is 64.5

## New Features set 6
    Additional fetures such as common prefixes of length 3 to 6 are included along with their normalised counts

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train[:,:19], y_train)   ## fitting only first four length based features

clf.predict(X_valid[:,:19])

clf.score(X_train[:,:19],y_train)

clf.score(X_valid[:,:19], y_valid)

y_pred=clf.predict(X_test[:,:19])

clf.score(X_test[:,:19], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train[:,:19], y_train)

clf.predict(X_valid[:,:19])

clf.score(X_train[:,:19],y_train)

clf.score(X_valid[:,:19], y_valid)

y_pred=clf.predict(X_test[:,:19])

clf.score(X_test[:,:19], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train[:,:19], y_train)

clf.predict(X_valid[:,:19])

clf.score(X_train[:,:19],y_train)

clf.score(X_valid[:,:19], y_valid)

y_pred=clf.predict(X_test[:,:19])

clf.score(X_test[:,:19], y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((71.45+71.87+72.07)/3)
avg_f1_scores.append((71.23+71.63+71.92)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 71.79 which is bit low compared to the paper's which is 73.2
    and f1_score of 71.59 is higher compared to paper's which is 64.8

## New Features Set 7
    Some miscellaneous features such as whether both questions contain 
    'not' whether they contain common digit or not

### Applying decision Tree
    All the parameters are as mentioned in the paper.
"""

clf = DecisionTreeClassifier(random_state=0, max_depth=10,min_samples_leaf=5)

clf = clf.fit(X_train, y_train)   ## fitting only first four length based features

clf.predict(X_valid)

clf.score(X_train,y_train)

clf.score(X_valid, y_valid)

y_pred=clf.predict(X_test)

clf.score(X_test, y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

dt_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Random Forest Classifier """

clf = RandomForestClassifier(n_estimators=50,max_depth=None, min_samples_leaf=5,random_state=0)
clf.fit(X_train, y_train)

clf.predict(X_valid)

clf.score(X_train,y_train)

clf.score(X_valid, y_valid)

y_pred=clf.predict(X_test)

clf.score(X_test, y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

rf_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

"""### Gradient boosted tree"""

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=4, random_state=0)

clf=clf.fit(X_train, y_train)

clf.predict(X_valid)

clf.score(X_train,y_train)

clf.score(X_valid, y_valid)

y_pred=clf.predict(X_test)

clf.score(X_test, y_test)

"""#### `F1-Score`"""

f1_score(y_test, y_pred, average='weighted')

"""#### Precision"""

precision_score(y_test, y_pred, average='weighted')

"""#### `Accuracy`"""

accuracy_score(y_test, y_pred)

gdb_acc=accuracy_score(y_test, y_pred)

"""#### Classification report """

report = classification_report(y_test, y_pred)
print(report)

avg_accuracies.append((71.62+72.47+72.50)/3)
avg_f1_scores.append((71.55+72.23+72.31)/3)

print(avg_accuracies)
print(avg_f1_scores)

"""#### Observation:
    Accuracy was 72.19 which is bit low compared to the paper's which is 74.6 and
    f1_score of 72.03 is higher compared to paper's which is 66.3

## Plotting Average accuracies and f1-scores of different feature sets
"""

plt.plot(avg_accuracies, label='Test Accuracies')
plt.plot(avg_f1_scores, label='Test F1-scores loss')
plt.title('Average accuracies and f1-scores using decision tree,Random forest and gradient boost trees ')
plt.legend();

from prettytable import PrettyTable
z = PrettyTable()

z.field_names = ["Feature Set", "Average  Accuracy%", "Average F-Scores"]
z.add_row(['L',avg_accuracies[0] ,avg_f1_scores[0]])
z.add_row(['L,LC',avg_accuracies[1] ,avg_f1_scores[1]])
z.add_row(['L,LC,LCXS',avg_accuracies[2] ,avg_f1_scores[2]])
z.add_row(['L,LC,LCXS,LW',avg_accuracies[3] ,avg_f1_scores[3]])
z.add_row(['L,LC,LCXS,LW,CAP',avg_accuracies[4] ,avg_f1_scores[4]])
z.add_row(['L,LC,LCXS,LW,CAP,PRE',avg_accuracies[5] ,avg_f1_scores[5]])
z.add_row(['L,LC,LCXS,LW,CAP,PRE,Misc',avg_accuracies[6] ,avg_f1_scores[6]])

print(z)

"""# Neural Network Models 

"""

import pickle
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Bidirectional,Activation, BatchNormalization,Flatten,Multiply
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from keras.preprocessing import text
from keras.utils import np_utils
from gensim.models import Word2Vec, KeyedVectors 
import gensim

nltk.download('punkt')
nltk.download('wordnet')

"""#### Downloading the GLove 840B 300 dimensions embeddings`
    
"""

### ' Don't run these cells they are already downloaded and embedding vector was extracted and saved in drive to save time each time.
### So, directly go to the file open embedding vector`

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/
!wget http://nlp.stanford.edu/data/glove.840B.300d.zip

"""##### `Path to save the embeddings and 300d.txt file `
    /content/drive/MyDrive/embedding_vector
"""

!unzip glove.840B.300d.zip

embedding_vector = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_vector[word] = coef

with open('embedding_vector', 'wb') as fp:
    pickle.dump(embedding_vector, fp)

"""### Opening already extracted embedding index"""

file=open('/content/drive/MyDrive/embedding_vector','rb')
embedding_vector=pickle.load(file)

embedding_vector

"""## LSTM
    Reference : https://towardsdatascience.com/finding-similar-quora-questions-with-word2vec-and-xgboost-1a19ad272c0d
    https://erogol.com/duplicate-question-detection-deep-learning/
    https://www.kaggle.com/nkaps98/quora-question-pairs-glove-lstm
    https://github.com/aerdem4/kaggle-quora-dup/blob/master/model.py


"""

def lemmatize(s):
    lemma = []
    wnl = WordNetLemmatizer()
    for doc in s:
        tokens = [wnl.lemmatize(w) for w in doc]
        lemma.append(tokens)

    # Removing Stopwords
    filter_words = []
    Stopwords = set(stopwords.words('english'))

    #ab = spell('nd')
    for sent in lemma:
        tokens = [w for w in sent if w not in Stopwords]
        filter_words.append(tokens)

    space = ' ' 
    sentences = []
    for sentence in filter_words:
        sentences.append(space.join(sentence))
        
    return sentences

train_que1= list(preprocessed_question1[:int(len(preprocessed_question1)*0.80)])
train_que2= list(preprocessed_question2[:int(len(preprocessed_question2)*0.80)])

test_que1= list(preprocessed_question1[int(len(preprocessed_question1)*0.80):])
test_que2= list(preprocessed_question2[int(len(preprocessed_question2)*0.80):])

questions = list(preprocessed_question1 + preprocessed_question2)

questions[:4]

### MAX_NB_WORDS is a constant which indicates
### the maximum number of words that should be present
MAX_NB_WORDS = 200000
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)

###### converting into sequences and adding padding if the sequence is less than 
###### max length = 30 
train_que1 = tokenizer.texts_to_sequences(train_que1)
train_que1 = pad_sequences(train_que1, maxlen = 30, padding='post')

train_que2 = tokenizer.texts_to_sequences(train_que2)
train_que2 = pad_sequences(train_que2, maxlen = 30, padding='post')

###### converting into sequences and adding padding if the sequence is less than 
###### max length = 30 
test_que1 = tokenizer.texts_to_sequences(test_que1)
test_que1 = pad_sequences(test_que1, maxlen = 30, padding='post')

test_que2 = tokenizer.texts_to_sequences(test_que2)
test_que2 = pad_sequences(test_que2, maxlen = 30, padding='post')

print(len(train_que1))
print(len(train_que2))

print(len(test_que1))
print(len(test_que2))

y_train = labels[:int(len(train_que1))]
y_test  = labels[int(len(train_que1)):]

print(len(y_train))
print(len(y_test))

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index)+1

print(vocab_size)

##### Preparing the embedding matrix that needs to be passed to the sequential model

embedding_matrix = np.zeros((len(word_index)+1, 300))
for word, i in word_index.items():
    emb_vect = embedding_vector.get(word)
    if emb_vect is not None:
        embedding_matrix[i] = emb_vect

difference_matrix = train_que1-train_que2

hadamard_matrix =  train_que1*train_que2

print(difference_matrix.shape)
print(hadamard_matrix.shape)
print(train_que1.shape)
print(train_que2.shape)
print(test_que1.shape)
print(test_que2.shape)

#####  Model for training question 1 #######
model_q1 = Sequential()
model_q1.add(Embedding(input_dim = len(word_index)+1,
                       output_dim = 300,
                      weights = [embedding_matrix],
                      input_length = 30))
model_q1.add(LSTM(128, activation = 'tanh', return_sequences = True))
model_q1.add(Dropout(0.2))
model_q1.add(LSTM(128, return_sequences = True))
model_q1.add(LSTM(128))
model_q1.add(Dense(60, activation = 'tanh'))
model_q1.add(Dense(2, activation = 'softmax'))

#####  Model for training question 2 #######
model_q2 = Sequential()
model_q2.add(Embedding(input_dim = len(word_index)+1,
                       output_dim = 300,
                      weights = [embedding_matrix],
                      input_length = 30))
model_q2.add(LSTM(128, activation = 'tanh', return_sequences = True))
model_q2.add(Dropout(0.2))
model_q2.add(LSTM(128, return_sequences = True))
model_q2.add(LSTM(128))
model_q2.add(Dense(60, activation = 'tanh'))
model_q2.add(Dense(2, activation = 'softmax'))

# Merging the output of the two models,i.e, model_q1 and model_q2
mergedOut = Multiply()([model_q1.output, model_q2.output])

mergedOut = Flatten()(mergedOut)
mergedOut = Dense(100, activation = 'relu')(mergedOut)
mergedOut = Dropout(0.2)(mergedOut)
mergedOut = Dense(50, activation = 'relu')(mergedOut)
mergedOut = Dropout(0.2)(mergedOut)
mergedOut = Dense(2, activation = 'softmax')(mergedOut)

new_model = Model([model_q1.input, model_q2.input], mergedOut)
new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'])

history = new_model.fit([train_que1,train_que2],y_train, batch_size = 1000, epochs = 10)

y_pred = new_model.predict([test_que1, test_que2], batch_size=2000, verbose=1)

prediction = []
for i in y_pred:
  if i[0]>i[1]:
    prediction.append(0)
  else:
    prediction.append(1)

#y_test
from sklearn.metrics import accuracy_score,f1_score
test_accuracy = accuracy_score(y_test, prediction, normalize=True)

print(f'Test accuracy using LSTM is {test_accuracy*100}')

F1_score= f1_score(y_test, prediction, average='weighted')

print(f'Test F1_score using LSTM is {F1_score*100}')

"""## CBOW"""

embedding_vector   ## sample

"""#### In the paper it was clearly mentioned that the vector representation of each question is sum of embeddings of words of that question """

embedding_vector['no'].shape   ### getting shape of a random sample

#########################  For preprocessed question1 #############################

que1 = [nltk.word_tokenize(sent) for sent in preprocessed_question1]

import time
start= time.time()

q1_feats = np.zeros((len(que1),300))
for i,question in enumerate(que1):
    #token.fit_on_texts(question)
    #seq = token.texts_to_sequences(question)
    #pad_seq = pad_sequences(seq,maxlen=300)
    
    sent_matrix=np.zeros((1,300),dtype='float32')
    for j,word in enumerate(question):
      embedding_value = embedding_vector.get(word)

      if embedding_value is not None:
          embedding_value=embedding_value.reshape(1,300)
          sent_matrix=np.add(sent_matrix, embedding_value,dtype='float32')
    
    if i% 20000 == 0:
      print(f'time taken for {i} data points is {time.time()-start}')

    q1_feats[i] = sent_matrix

q1_feats.shape

#########################  For preprocessed question2 #############################

que2 = [nltk.word_tokenize(sent) for sent in preprocessed_question2]

import time
start= time.time()

q2_feats = np.zeros((len(que2),300))
for i,question in enumerate(que2):
    #token.fit_on_texts(question)
    #seq = token.texts_to_sequences(question)
    #pad_seq = pad_sequences(seq,maxlen=300)
    
    sent_matrix=np.zeros((1,300),dtype='float32')
    for j,word in enumerate(question):
      embedding_value = embedding_vector.get(word)

      if embedding_value is not None:
          embedding_value=embedding_value.reshape(1,300)
          sent_matrix=np.add(sent_matrix, embedding_value,dtype='float32')
    
    if i% 20000 == 0:
      print(f'time taken for {i} data points is {time.time()-start}')

    q2_feats[i] = sent_matrix

q2_feats.shape

del que1
del que2
del embedding_vector

"""#### Preparing concatenated matrix"""

diff_feats = q1_feats - q2_feats
hadamard_feats = q1_feats * q2_feats

feat1 = np.hstack((q1_feats,q2_feats))

del q1_feats 
del q2_feats

feat2 = np.hstack((diff_feats,hadamard_feats))

del diff_feats 
del hadamard_feats

features = np.hstack((feat1,feat2))

del feat1
del feat2

"""#### Defining the model
    3 Layer MLP followed by softmax
"""

model = Sequential()
model.add(Dense(200,input_dim=1200))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

y=labels
X=features

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape

X_test.shape

with open('CBOW_train', 'wb') as fp:
    pickle.dump(X_train, fp)

with open('CBOW_test', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('CBOW_train_label', 'wb') as fp:
    pickle.dump(y_train, fp)

with open('CBOW_test_label', 'wb') as fp:
    pickle.dump(y_test, fp)

del features 
del X

del preprocessed_question1
del preprocessed_question2

model.fit(X_train,y_train,batch_size=2000,epochs=40)

predictions = model.predict(X_test,batch_size=200,verbose=1)

predictions

y_pred=[]
for i in predictions:
  if i[0]>i[1]:
    y_pred.append(0)
  else:
    y_pred.append(1)

#y_test
from sklearn.metrics import accuracy_score,f1_score
test_accuracy = accuracy_score(y_test, y_pred, normalize=True)

print(f'Test accuracy using continuous bag of words is {test_accuracy*100}')

F1_score= f1_score(y_test, y_pred, average='weighted')

print(f'Test F1_score using continuous bag of words is {F1_score*100}')



"""## References:


1.    https://www.kaggle.com/c/quora-question-pairs
2.   https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30
3. https://towardsdatascience.com/finding-similar-quora-questions-with-bow-tfidf-and-random-forest-c54ad88d1370
4.  https://towardsdatascience.com/finding-similar-quora-questions-with-word2vec-and-xgboost-1a19ad272c0d
5. https://erogol.com/duplicate-question-detection-deep-learning/
6. https://www.kaggle.com/nkaps98/quora-question-pairs-glove-lstm
7. https://github.com/aerdem4/kaggle-quora-dup/blob/master/model.py


"""