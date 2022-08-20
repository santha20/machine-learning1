#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv('train.csv')


# In[4]:


# printing the first 5 rows of the dataframe
titanic_data.head()


# In[5]:


# number of rows and Columns
titanic_data.shape


# In[6]:


# getting some informations about the data
titanic_data.info()


# In[7]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# In[8]:


# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[9]:


# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[10]:


# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())


# In[11]:


print(titanic_data['Embarked'].mode()[0])


# In[12]:


# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[13]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# In[14]:


# getting some statistical measures about the data
titanic_data.describe()


# In[15]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# In[16]:


sns.set()


# In[17]:


# making a count plot for "Survived" column
sns.countplot('Survived', data=titanic_data)


# In[18]:


titanic_data['Sex'].value_counts()


# In[19]:


# making a count plot for "Sex" column
sns.countplot('Sex', data=titanic_data)


# In[20]:


# number of survivors Gender wise
sns.countplot('Sex', hue='Survived', data=titanic_data)


# In[21]:


# making a count plot for "Pclass" column
sns.countplot('Pclass', data=titanic_data)


# In[22]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)


# In[23]:


titanic_data['Sex'].value_counts()


# In[24]:


titanic_data['Embarked'].value_counts()


# In[25]:


# converting categorical Columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[26]:


titanic_data.head()


# In[27]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[28]:


print(X)


# In[29]:


print(Y)


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[31]:


print(X.shape, X_train.shape, X_test.shape)


# In[32]:


model = LogisticRegression()


# In[33]:


# training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# In[34]:


# accuracy on training data
X_train_prediction = model.predict(X_train)


# In[35]:


print(X_train_prediction)


# In[36]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[37]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[38]:


print(X_test_prediction)


# In[39]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




