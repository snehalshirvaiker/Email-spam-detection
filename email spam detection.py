#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


spam_df = pd.read_csv("emails.csv")


# In[3]:


spam_df.head(10)


# In[4]:


spam_df.tail()


# In[5]:


spam_df.describe()


# In[6]:


spam_df.info()


# In[7]:


ham = spam_df[spam_df['spam']==0]


# In[8]:


spam = spam_df[spam_df['spam']==1]


# In[9]:


ham


# In[10]:


spam


# In[11]:


print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")


# In[12]:


print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")


# In[13]:


sns.countplot(spam_df['spam'], label = "Count") 


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


# In[18]:


print(vectorizer.get_feature_names())


# In[19]:


print(spamham_countvectorizer.toarray())  


# In[20]:


spamham_countvectorizer.shape


# In[21]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spamham_countvectorizer, label)


# In[22]:


testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[23]:


test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[ ]:


# Mini Challenge!
testing_sample = ['Hello, I am Ryan, I would like to book a hotel in Bali by January 24th', 'money viagara!!!!!']


# In[38]:


testing_sample = ['money viagara!!!!!', "Hello, I am Ryan, I would like to book a hotel in SF by January 24th"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[25]:


X = spamham_countvectorizer
y = label


# In[26]:


X.shape


# In[27]:


y.shape


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[29]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[30]:


# from sklearn.naive_bayes import GaussianNB 
# NB_classifier = GaussianNB()
# NB_classifier.fit(X_train, y_train)


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix


# In[32]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[33]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[34]:


print(classification_report(y_test, y_predict_test))

