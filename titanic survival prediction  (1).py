#!/usr/bin/env python
# coding: utf-8

# In[ ]:


CODSOFT INTERNSHIP TASK 1
   TITANIC SURVIVAL PREDICTION
NAME : SOWMIYA R
    


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("archive1.csv")
df.head()


# In[3]:


df.head(12)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df['Survived'].value_counts()


# In[7]:


sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[8]:


df["Sex"]


# In[9]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[10]:


df.groupby('Sex')[['Survived']].mean()


# In[11]:


df['Sex'].unique()


# In[15]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex'] = labelencoder.fit_transform(df['Sex'])
df.head()


# In[16]:


df['Sex'],df['Survived']


# In[17]:


sns.countplot(x=df['Pclass'], hue=df["Survived"])


# In[18]:


df.isna().sum()


# In[19]:


df=df.drop(['Age'],axis=1)


# In[20]:


df_final=df
df_final.head(10)


# In[21]:


X=df[['Pclass','Sex']]
Y=df['Survived']


# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=0)


# In[24]:


from sklearn.linear_model import LogisticRegression
log= LogisticRegression(random_state=0)
log.fit(X_train,Y_train)


# In[ ]:


#unable to run logistic regression in jupyter notebook 


# In[25]:


p= print(log.predict(X_test))


# In[26]:


print(Y_test)


# In[28]:


import warnings
warnings.filterwarnings("ignore")
r=log.predict([[2,1]])
if(r==0):
    print("Not survived")
else:
    print("survived")
    


# In[ ]:




