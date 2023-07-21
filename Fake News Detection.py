#!/usr/bin/env python
# coding: utf-8

"""
Created on Sun July 15 2023

@author: Nada Osama
"""

# ![header.jpg](attachment:header.jpg)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from simple_colors import *


# In[2]:


data = pd.read_csv('news data.csv')


# In[3]:


data.head()


# In[4]:


df = data.copy()


# ***

# # <span style='background :#10e2ee' > Data Preprocessing </span>
# - Removing the Null values
# - Adding a new field
# - Drop features that are not needed
# - Text Processing

# #### <span style='background :#10e2ee' > Removing the Null values </span>

# In[5]:


df.isnull().sum()


# In[6]:


df['Body'] = df['Body'].fillna('')


# In[7]:


df.isnull().sum()


# #### <span style='background :#10e2ee' > Adding a new field </span>

# In[8]:


df['News'] = df['Headline'] + df['Body']


# In[9]:


df.head()


# #### <span style='background :#10e2ee' > Drop features that are not needed </span>

# In[10]:


useless_cols = ['URLs', 'Headline', 'Body']
df = df.drop(useless_cols, axis = 1)


# In[11]:


df.columns


# #### <span style='background :#10e2ee' > Text Processing </span>

# In[12]:


# removing symbols, and stopwords

p = PorterStemmer()
def cleaning(data):
    data = re.sub('[^a-zA-Z]', ' ', data)
    data = data.lower()
    data = data.split()
    data = [p.stem(word) for word in data if not word in stopwords.words('english')]
    data = ' '.join(data)
    return data


# In[13]:


df['News'] = df['News'].apply(cleaning)


# In[14]:


df.head()


# ***

# # <span style='background :#10e2ee' > Splitting the dataset </span>

# In[15]:


x = df['News']
y = df['Label']

# training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# In[16]:


vect = TfidfVectorizer()
xv_train = vect.fit_transform(x_train)
xv_test = vect.transform(x_test)


# ***

# # <span style='background :#10e2ee' > Model Fitting </span>
# - SVM
# - Logistic Regression
# 

# #### <span style='background :#10e2ee' > SVM </span>
# 

# In[17]:


SVM_model = SVC(kernel = 'linear')

# Fitting  
SVM_model.fit(xv_train, y_train)

# Predicting  
SVM_y_pred = SVM_model.predict(xv_test)

# Calculating the accuracy 
score = accuracy_score(y_test, SVM_y_pred)
print('Accuracy of SVM model is: %0.4f ' % score)

# plotting confusion matrix
cm = metrics.confusion_matrix(y_test, SVM_y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.RdBu)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
ax.yaxis.set_ticklabels(['FAKE', 'REAL']);


# In[18]:


print(classification_report(y_test, SVM_y_pred))


# #### <span style='background :#10e2ee' > Logistic Regression </span>

# In[19]:


LR_model = LogisticRegression()

# Fitting 
LR_model.fit(xv_train, y_train)

# Predicting 
LR_y_pred = LR_model.predict(xv_test)

# Calculating the accurracy 
score = accuracy_score(y_test, LR_y_pred)
print('Accuracy of LR model is: %0.4f ' % score)

# plotting confusion matrix
cm = metrics.confusion_matrix(y_test, LR_y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.RdBu)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
ax.yaxis.set_ticklabels(['FAKE', 'REAL']);


# In[20]:


print(classification_report(y_test, LR_y_pred))


# ***

# # <span style='background :#10e2ee' > Model Testing (SVM model) </span>

# In[21]:


def fake_news_det(news):
    input_data = {'text':[news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test['text'] = new_def_test['text'].apply(cleaning) 
    new_x_test = new_def_test['text']
    vectorized_input_data = vect.transform(new_x_test)
    prediction = SVM_model.predict(vectorized_input_data)
    
    if prediction == 1:
        print(green('Not Fake News'))
    else:
        print(red('Fake News'))


# In[22]:


print(black('Test 1 \n', ['bold']))
news = str(input())
print('\n')
fake_news_det(news)


# In[23]:


print(black('Test 2 \n', ['bold']))
news = str(input())
print('\n')
fake_news_det(news)


# In[24]:


print(black('Test 3 \n', ['bold']))
news = str(input())
print('\n')
fake_news_det(news)


# In[25]:


print(black('Test 4 \n', ['bold']))
news = str(input())
print('\n')
fake_news_det(news)


# ***
