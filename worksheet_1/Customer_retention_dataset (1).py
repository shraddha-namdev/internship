#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[4]:


train_df = pd.read_excel('customer_retention_code.xlsx')
train_df


# In[ ]:





# In[5]:


train_df.shape


# In[6]:


train_df.sample(5)


# In[7]:


train_df.isnull().sum()


# In[8]:


train_df.describe()


# In[9]:


train_df.corr()


# In[10]:


categorical_col,numeric_col=[],[]
for i in train_df:
    if train_df[i].dtype=='O':
        categorical_col.append(i)
    elif (train_df[i].dtypes=='int64') | (train_df[i].dtypes=='float64') | (train_df[i].dtypes=='int32'):
        numeric_col.append(i)
    else: continue
print('>>> Total Number of Feature::', train_df.shape[1])
print('>>> Number of categorical features::', len(categorical_col))
print('>>> Number of Numerical Feature::', len(numeric_col))


# In[11]:


train_df.hist(figsize=(30,28));


# In[12]:


plt.figure(figsize=(10,9))
sns.countplot(train_df['1Gender of respondent'])
plt.xticks(rotation=90)


# In[ ]:





# In[13]:


sns.countplot(x="1Gender of respondent", data=train_df)


# In[14]:


for i in train_df.columns:
    sns.countplot(train_df[i])
    plt.show()
          


# In[15]:


train_df.hist(bins=25,figsize=(100,90))
# display histogram
plt.show()


# In[16]:


# ploting heatmap
import seaborn
import seaborn as sb
plt.figure(figsize=[19,10],facecolor='yellow')
sb.heatmap(train_df.corr(),annot=True)


# In[17]:


for a in range(len(train_df.corr().columns)):
    for b in range(a):
        if abs(train_df.corr().iloc[a,b]) >0.7:
            name = train_df.corr().columns[a]
            print(name)


# In[18]:


new_df=train_df.drop('3 Which city do you shop online from?',axis=1)
new_df


# In[19]:


new_df.update(new_df.fillna(new_df.mean()))


# In[20]:


# catogerical vars 
import pandas as pd
next_df = pd.get_dummies(new_df,drop_first=True)
# display new dataframe
next_df


# In[21]:


next_df.plot(kind='box',subplots=True,layout=(120,15))


# # Splitting dataset
# 

# In[22]:


y=next_df.iloc[:,-1]


# In[23]:


y.head()


# In[24]:


y.shape


# In[25]:


x=next_df.iloc[:,0:8]
x


# In[26]:


x.shape


# In[27]:


y.shape


# In[80]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.22,random_state=40)


# In[81]:


x_train.shape


# In[82]:


x_test.shape


# In[83]:


y_train.shape


# In[84]:


new_df


# In[85]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[86]:


x


# In[87]:


y


# In[88]:



reg = LinearRegression()
reg


# In[ ]:





# In[89]:


reg.score(x, y)


# In[90]:


reg.coef_


# In[91]:


reg.intercept_


# In[92]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[93]:


X, y = make_classification(n_samples=233, n_features=8,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[94]:


clf.fit(x, y)


# In[98]:


print(clf.predict([[0, 0, 0, 0]]))


# In[96]:


clf.fit(x_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




