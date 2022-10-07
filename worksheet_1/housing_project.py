#!/usr/bin/env python
# coding: utf-8

# In[1]:


#EDA ON  HOUSE PRICES


# In[85]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import scipy as sts
from scipy import stats
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

import warnings
warnings.filterwarnings('ignore')


# In[86]:


train = pd.read_excel('housing_train.xlsx')
train


# In[87]:


test = pd.read_excel('house_test.xlsx')
test


# In[88]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[89]:


train.describe()


# In[90]:


test.describe()


# In[91]:


train.isnull().sum()


# In[92]:


test.isnull().sum()


# In[93]:


train.info()


# In[94]:


test.info()


# In[95]:


train.shape


# In[96]:


test.shape


# In[97]:


train.columns


# In[98]:


test.columns


# In[99]:


train.dtypes


# In[100]:


train.corr()


# In[101]:


test.corr()


# In[ ]:





# In[ ]:





# In[102]:


Col_with_na=[nan_col for nan_col in train.columns if train[nan_col].isnull().sum()>1]

## 2- step print the feature name and the percentage of missing values

for nan_col in Col_with_na:
    print(nan_col, np.round(train[nan_col].isnull().mean(), 4),  ' % missing values')


# In[103]:


Col_with_na=[nan_col for nan_col in test.columns if test[nan_col].isnull().sum()>1]


for nan_col in Col_with_na:                                                                 ## printing the column name and the percentage of missing values

    print(nan_col, np.round(test[nan_col].isnull().mean(), 4),  ' % missing values')


# In[104]:


for i in train.describe().columns:
    sns.distplot(train[i])
    plt.xticks(rotation=90)
    plt.show()


# In[105]:


for i in test.describe().columns:
    sns.distplot(test[i])
    plt.xticks(rotation=90)
    plt.show()


# In[106]:


train.MSZoning.value_counts()


# In[107]:


train.Street.value_counts()


# In[108]:


train.Alley.value_counts()


# In[109]:


train.LotShape.value_counts()


# In[110]:


train.LandContour.value_counts()


# In[111]:


train.Utilities.value_counts()


# In[112]:


train.Fence.value_counts()


# In[113]:


train.PoolQC.value_counts()


# In[114]:


train.YrSold.value_counts()


# In[115]:


train.SaleCondition.value_counts()


# In[116]:


train.describe()

####################### visulaizaion##########
# In[117]:


train


# In[118]:


test


# In[119]:


for i in train.describe().columns:
    # Train[i].plot.box()
    sns.boxplot(train[i])
    plt.xticks(rotation=90)
    plt.show()


# In[120]:


plt.figure(figsize=(32,32))
sns.heatmap(train.corr(),annot=True,fmt='.0%',cmap='viridis')
plt.title('Correlation between  different attributes')

plt.show()


# In[121]:


train.hist(figsize=(30,28));


# In[122]:


test.hist(figsize=(30,28));


# In[123]:


train.boxplot(figsize=(12,12));
plt.xticks(rotation=90)


# In[124]:


plt.barh(train['SaleCondition'],train['SalePrice'])
plt.title('SaleCondition and Price')
plt.xlabel('SalePrice')
plt.ylabel('SaleCondition')
plt.show()


# In[125]:


plt.barh(train['LandContour'],train['SalePrice'])
plt.title('LandContour and Price')
plt.xlabel('SalePrice')
plt.ylabel('LandContour')
plt.show()


# In[126]:


plt.barh(train['YrSold'],train['SalePrice'])
plt.title('YrSold and Price')
plt.xlabel('SalePrice')
plt.ylabel('YrSold')
plt.show()


# In[127]:


plt.figure(figsize=(10,9))
sns.countplot(train.Street)
plt.xticks(rotation=90)


# In[128]:


plt.figure(figsize=(10,9))
sns.countplot(train.MSZoning)
plt.xticks(rotation=90)


# In[129]:


plt.figure(figsize=(13,9))
plt.xlabel('SalePrice')
plt.ylabel('LotArea')
plt.title('SalePrice and LotArea')
sns.scatterplot(x='SalePrice',y='LotArea',hue='MSZoning',size='SalePrice',data=train)


# In[130]:


train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[131]:


#Factor plot of OverallQual vs SalePrice
sns.factorplot(x="OverallQual",y="SalePrice",data=train,kind="bar",size = 4,palette = "muted",aspect=3)
plt.title('Price of the house according to rating of ther material and finishing',fontsize=25)
plt.ylabel("Sale Price of the house")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(train['OverallQual'],palette= 'rainbow')
plt.title("Rating of the overall material and finish of the house",fontsize=20)
plt.xticks(rotation=90)
plt.show()

print(train.OverallQual.value_counts())


# In[ ]:


#Checking the correlation with target variable that is SalePrice
plt.figure(figsize=(18,8))
train.drop('SalePrice', axis=1).corrwith(train['SalePrice']).plot(kind='bar',grid=True )
plt.xticks(rotation=90)
plt.title("Correlation with target Variable",fontsize=25)
plt.show()


# In[132]:


df=pd.concat([train,test],ignore_index=True)
df


# In[133]:


#Seeing the NaN values in ratio
print('\nPercentage of missing values in each column:')
(df.isnull().sum() / df.shape[0]).sort_values(ascending=False).head(20)


# In[138]:


#df = df.drop('PoolQC', axis=1)
df = df.drop('MiscFeature', axis=1)
df = df.drop('Alley', axis=1)
df = df.drop('Fence',axis=1)
df = df.drop('FireplaceQu',axis=1)


# In[139]:


df.shape


# In[140]:


#Number of missing values and their percentage in each row
print(pd.DataFrame.from_dict({'Rows' : df.isnull().any(axis = 1), 
                              'missing values' : df.isnull().sum(axis = 1), 
                              'Percentage of missing values' : round(100*df.isnull().sum(axis = 1)/df.shape[1])})
     )


# In[141]:


#Number and percentage of rows having more than 1 missing values
# count the number of rows having > 1 missing values
print("Number of rows having more than 5 missing values : ", len(df[df.isnull().sum(axis=1) > 1].index))
print("Number of rows having more than 5 missing values : ", 100*(len(df[df.isnull().sum(axis=1) > 1].index) / len(df.index)))


# In[142]:


#Number and percentage of rows having more than 1 missing values
# count the number of rows having > 1 missing values
print("Number of rows having more than 5 missing values : ", len(df[df.isnull().sum(axis=1) > 1].index))
print("Number of rows having more than 5 missing values : ", 100*(len(df[df.isnull().sum(axis=1) > 1].index) / len(df.index)))


# In[143]:


#Percentage of missing values in each column
print(round(100*(df.isnull().sum()/len(df.index)), 2))


# In[144]:


df['LotFrontage'].describe(percentiles = [0.9, 0.95, 0.99])


# In[146]:


df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)


# In[147]:


df.describe()


# In[150]:


df


# In[151]:


df_corelation = df.corr()
df_corelation


# In[154]:


plt.figure(figsize=(32,32))
sns.heatmap(train.corr(),annot=True,fmt='.0%',cmap='viridis')
plt.title('Correlation between  different attributes')


# In[158]:


plt.show()


# In[159]:


df['SalePrice'].fillna(df['SalePrice'].mean(),inplace=True)


# In[160]:


categorical_features=[col for col in df.columns if df[col].dtype=='O']


# In[161]:


categorical_features


# In[162]:


for i in categorical_features:
    temp=df.groupby(i)['SalePrice'].count()/len(df)
    temp_df=temp[temp>0.01].index
    df[i]=np.where(df[i].isin(temp_df),df[i],'Rare_var')


# In[163]:


df.head()


# In[164]:


for i in categorical_features:
    labels_ordered=df.groupby([i])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[i]=df[i].map(labels_ordered)


# In[165]:


y = df['SalePrice']
x = df.drop('SalePrice', axis=1)


# In[166]:


print(x.shape, y.shape)


# In[167]:


print(x.shape, y.shape)


# In[168]:


x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
x.shape


# In[169]:


x


# In[170]:


y


# In[175]:


#Using GridSearchCV to find out the best parameter in Lasso

alpha_value={'alpha':[.1,.01,.001,.0001,.0]}
model_test= Lasso()
grid=GridSearchCV(estimator=model_test, param_grid=alpha_value)


# In[173]:


grid.fit(x_train,y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
print(grid.best_params_)
print('/n')


# In[ ]:


#Using GridSearchCV to find out the best parameter in Ridge

alpha_value1={'alpha':[.1,.01,.001,.0001,0]}
model_test1= Ridge()
grid=GridSearchCV(estimator=model_test1, param_grid=alpha_value1)


# In[174]:


grid.fit(x_train,y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
print(grid.best_params_)
print('/n')


# In[ ]:


#Using GridSearchCV to find out the best parameters in RandomForestRegressor

esti_n={'n_estimators':[10,50,100,200]}
model_test2=RandomForestRegressor()
grid=GridSearchCV(model_test2,param_grid=esti_n)


# In[ ]:


grid.fit(x_train,y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.n_estimators)
print(grid.best_params_)
print('/n')


# In[ ]:


#Using GridSearchCV to find out the best parameter in GradientBoostingRegressor

GBR_Esti={'n_estimators':[10,100,200,300]}
model_test2=GradientBoostingRegressor()
grid=GridSearchCV(model_test2,param_grid=GBR_Esti)

grid.fit(x_train,y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.n_estimators)
print(grid.best_params_)
print('/n')


# In[ ]:


#Using GridSearchCV to find out the best parameters in ExtraTreesRegressor

ETR_Esti={'n_estimators':[10,50,100,200,150,250,300]}
model_test3=ExtraTreesRegressor()

grid=GridSearchCV(model_test2,param_grid=ETR_Esti)


# In[ ]:


grid.fit(x_train,y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.n_estimators)
print(grid.best_params_)
print('/n')


# In[ ]:


import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error,explained_variance_score,r2_score,mean_tweedie_deviance

import warnings
warnings.filterwarnings('ignore')

model=[LinearRegression(), Lasso(alpha=.1),Ridge(alpha=.1),KNeighborsRegressor(),DecisionTreeRegressor(),
      RandomForestRegressor(n_estimators=200),AdaBoostRegressor(),
       GradientBoostingRegressor(n_estimators=100),ExtraTreesRegressor(n_estimators=50)]


for m in model:
    m.fit(x_train,y_train)
    m.score(x_train,y_train)
    score=m.score(x_train,y_train)

    predm=m.predict(x_test)
    print(' Score = \n',m,'is :',score )
    print('R2_Score',r2_score(y_test,predm))
    print('EVS',explained_variance_score(y_test,predm))
    print('MAE',mean_absolute_error(y_test,predm))
    print('MSE',mean_squared_error(y_test,predm))
    print('MTD',mean_tweedie_deviance(y_test,predm))
    print('R_MSE:',np.sqrt(mean_squared_error(y_test,predm)))

    print('\n')


# In[ ]:





# In[ ]:




