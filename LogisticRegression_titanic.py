#!/usr/bin/env python
# coding: utf-8

# ## Collecting Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math

titanic_data=pd.read_csv("C:\\Users\\Shafaq Murtaza\\Python Projects\\titanic.csv")
titanic_data.head(10)
print("Number of passengers in original data:"+str(len(titanic_data.index)))


# ## Analyzing Data

# In[2]:


sns.countplot(x="Survived", data=titanic_data)


# In[3]:


sns.countplot(x="Survived", hue="Sex", data=titanic_data)


# In[4]:


sns.countplot(x="Survived", hue="Pclass", data=titanic_data)


# In[5]:


titanic_data["Age"].plot.hist()


# In[6]:


titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))


# In[7]:


titanic_data.info()


# In[8]:


sns.countplot(x="SibSp", data=titanic_data)


# In[9]:


sns.countplot(x="Parch", data=titanic_data)


# ## Data Wrangling

# In[10]:


titanic_data.isnull()


# In[11]:


titanic_data.isnull().sum()


# In[12]:


sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap="viridis")


# In[13]:


sns.boxplot(x="Pclass", y="Age", data=titanic_data)


# In[14]:


titanic_data.head(5)


# In[15]:


titanic_data.drop("Cabin", axis=1, inplace=True)


# In[16]:


titanic_data.head(5)


# In[17]:


titanic_data.dropna(inplace=True)


# In[18]:


titanic_data.head(5)


# In[19]:


sns.heatmap(titanic_data.isnull(), yticklabels=False)


# In[20]:


titanic_data.isnull().sum()


# In[21]:


titanic_data.head(2)


# In[22]:


sex=pd.get_dummies(titanic_data['Sex'], drop_first=True)
sex.head(5)


# In[23]:


embark=pd.get_dummies(titanic_data['Embarked'], drop_first=True)
embark.head(5)


# In[24]:


pcl=pd.get_dummies(titanic_data['Pclass'], drop_first=True)
pcl.head(5)


# In[25]:


titanic_data=pd.concat([titanic_data,sex,embark,pcl],axis=1)


# In[26]:


titanic_data.head(5)


# In[27]:


titanic_data.drop(['Sex', 'Embarked','PassengerId', 'Name','Ticket'],axis=1, inplace=True)


# In[28]:


titanic_data.head(5)


# In[29]:


titanic_data.drop(['Pclass'], axis=1, inplace=True)


# In[30]:


titanic_data.head(5)


# ## Train Data

# In[33]:


y=titanic_data["Survived"]
titanic_data.head(2)


# In[39]:


x=titanic_data.drop('Survived', axis=1)


# In[41]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


logmodel=LogisticRegression()


# In[47]:


logmodel.fit(X_train,y_train)


# In[48]:


predictions=logmodel.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report


# In[52]:


classification_report(y_test, predictions)


# In[53]:


from sklearn.metrics import confusion_matrix


# In[54]:


confusion_matrix(y_test, predictions)


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


accuracy_score(y_test, predictions)


# In[ ]:




