
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from nltk.stem import WordNetLemmatizer


# In[2]:

import os

#list the pos files
filelist_pos = os.listdir("F:\\Intern Assignments\\DataWeave\\txt_sentoken\\pos\\") 
(filelist_pos)


# In[3]:

#list the neg files
filelist_neg = os.listdir("F:\\Intern Assignments\\DataWeave\\txt_sentoken\\neg\\") 
(filelist_neg)


# In[4]:

#read them into pandas
pos_list=[]
for i in xrange(0,len(filelist_pos)):
    f=open("F:\\Intern Assignments\\DataWeave\\txt_sentoken\\pos\\"+filelist_pos[i])
    pos_list.append(f.read())


# In[5]:

for i in xrange(0,len(filelist_neg)):
    f=open("F:\\Intern Assignments\\DataWeave\\txt_sentoken\\neg\\"+filelist_neg[i])
    pos_list.append(f.read())


# In[6]:

pos_list[0]


# In[7]:

rev_df=pd.DataFrame({'Review_Text':pos_list})
rev_df


# In[8]:

rev_df['Sentiment']=0
rev_df['Sentiment'][0:1000]=1
rev_df['Sentiment'][1000:1999]=-1
rev_df[998:1001]


# In[9]:

real_data=rev_df['Review_Text']


# In[10]:

##from nltk import PorterStemmer
##[ PorterStemmer().stem_word(word) for word in rev_df['Review_Text'][0].split(" ")] 
##rev_df['stem_review']=[ PorterStemmer().stem_word(word) for word in rev_df['Review_Text'][0].split(" ")] 


# In[11]:

import nltk
import string
tokens=[]
for i in range(0,len(real_data)):
    tokens.append(nltk.word_tokenize(real_data[i]))
for i in range(0,len(tokens)):
    tokens[i]=[word for word in tokens[i] if not all(char in string.punctuation for char in word)]
    tokens[i]=[word for word in tokens[i] if not (word.startswith("'") and len(word)<=2)]


# In[12]:

tokens_new=[]
for i in range(0,len(tokens)):
    tokens_new.append(" ".join(tokens[i]))
tokens_new[0]


# In[13]:

rev_df['tokens_new']=tokens_new


# In[14]:

import random            
num_to_select = 100                          
list_of_random_items =rev_df.sample(n= num_to_select)
list_of_random_items


# In[26]:

test_df=pd.DataFrame({'Review_Text':list_of_random_items['tokens_new']})
test_df['Actual_Sentiment']=list_of_random_items['Sentiment']
test_df


# In[20]:

import re
import sklearn
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# In[17]:

corpus_train = rev_df['Review_Text']
vectorizer_train = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , token_pattern=r'\w+' , sublinear_tf=False)
tfidf_train=vectorizer_train.fit_transform(corpus_train).todense()
corpus_test = test_df['Review_Text']
vectorizer_test = TfidfVectorizer(stop_words='english')
tfidf_test=vectorizer_train.transform(corpus_test)


# In[18]:

predictors_train = tfidf_train
predictors_test = tfidf_test
targets_train = rev_df['Sentiment']


# In[21]:

clf_svm= LinearSVC()
classifier_svm=clf_svm.fit(predictors_train,targets_train)
predictions_svm=classifier_svm.predict(predictors_test)


# In[27]:

test_df['Computed_Sentiment'] = predictions_svm
test_df


# In[30]:

test_df[['Review_Text','Actual_Sentiment','Computed_Sentiment']].to_csv("F:\Intern Assignments\DataWeave/MoviewReview.csv")


# In[ ]:



