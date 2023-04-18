#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Reading the CSV file containing the dataset using pd.read_csv(). The first column is set as the index of the DataFrame.
df = pd.read_csv('path',index_col=0)


# In[ ]:


#Displaying the first few rows of the DataFrame
df.head()


# In[5]:


#printing out the column names of the DataFrame
df.columns


# ## DATA CLEANING E IMPUTING

# In[10]:


#Creating a heatmap of the correlation matrix for the DataFrame
plt.figure(figsize = (70,40))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


#Dropping a list unnecessary columns
df.drop(['Unnamed: 0','ID_CLIE','FL_SOTT_BFP',
       'FL_SOTT_FONDI', 'FL_SOTT_MULTIRAMO', 'FL_SOTT_PREVIDENZA',
       'FL_SOTT_PROTEZIONE', 'FL_SOTT_RAMO_I', 'FL_SOTT_SUPERSMART',
       'FL_SOTT_TITOLI', 'SOTT_BFP_IMP', 'SOTT_FONDI_IMP',
       'SOTT_MULTIRAMO_IMP', 'SOTT_PREVIDENZA_IMP', 'SOTT_PROTEZIONE_IMP',
       'SOTT_RAMO_I_IMP', 'SOTT_SUPERSMART_IMP', 'SOTT_TITOLI_IMP',
       'FL_RACCOLTA_2022'], inplace=True, axis=1)


# In[ ]:


#Printing out information about the DataFrame using df.info()
#in order to check for missing values
print(df.info()) 

#Printing out the number of rows in the DataFrame using df.shape[0].
print(df.shape[0])

#Calculating the percentage of missing values in the DataFrame using df.isna().sum().sum().
na_vals = df.isna().sum().sum()
tot_vals = df.shape[0] * df.shape[1]
na_percentage = na_vals / tot_vals
print(f'{(na_percentage * 100):.2f}% of cells are missing values.')


# In[ ]:


#Printing out the unique values in the 'DS_CLUSTER_DIGITAL' column of the DataFrame using df['DS_CLUSTER_DIGITAL'].unique()
df.DS_CLUSTER_DIGITAL.unique()


# In[ ]:


#Replacing the string 'Non applicabile' with np.nan in the 'DS_CLUSTER_DIGITAL' column of the DataFrame using df['DS_CLUSTER_DIGITAL'].replace()
df['DS_CLUSTER_DIGITAL'].replace('Non applicabile', np.nan, inplace=True)
print(df['DS_CLUSTER_DIGITAL'].unique())


# In[ ]:


#Printing out the number of rows in the DataFrame using df.shape[0].
print(df.shape[0])
#Calculating the percentage of missing values in the DataFrame using df.isna().sum().sum().
na_vals = df.isna().sum().sum()
tot_vals = df.shape[0] * df.shape[1]
na_percentage = na_vals / tot_vals
print(f'{(na_percentage * 100):.2f}% of cells are missing values.')
#Printing out the number of missing values in the DataFrame using na_vals
print(na_vals)


# In[ ]:


#Printing out information about the DataFrame using df.info().
print(df.info()) 


# In[ ]:


#Importing SimpleImputer from sklearn.impute.
from sklearn.impute import SimpleImputer
#Creating a new DataFrame dfmode that includes only the columns 'DS_PORTAFOGLIO', 'CLUSTER_DIGITAL', and 'DS_CLUSTER_DIGITAL'.
modecols = ['DS_PORTAFOGLIO','CLUSTER_DIGITAL','DS_CLUSTER_DIGITAL']
dfmode=df[modecols]

#Creating an instance of SimpleImputer with the strategy set to 'most_frequent' and fitting it to dfmode using si.fit_transform().
si = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
#Creating a new DataFrame imputed from the imputed values with column names obtained from si.get_feature_names_out().
imputed = pd.DataFrame(si.fit_transform(dfmode), columns = si.get_feature_names_out())


# In[ ]:


#Checking if all missing values where imputed correctly using imputed.info().
imputed.info()


# In[ ]:


#Dropping the columns 'DS_PORTAFOGLIO', 'CLUSTER_DIGITAL', and 'DS_CLUSTER_DIGITAL' from the original DataFrame df.
df.drop(columns=['DS_PORTAFOGLIO','CLUSTER_DIGITAL','DS_CLUSTER_DIGITAL'],axis=1,inplace=True)


# In[ ]:


#Merging df with imputed on their indices using pd.merge().
df1=pd.merge(df,imputed,right_index=True,left_index=True)


# In[ ]:


#Printing out the number of rows in the imputed DataFrame using imputed.shape[0] 
print(imputed.shape[0])

#Calculating the percentage of missing values in the merged DataFrame df1 using df1.isna().sum().sum(). 
na_vals = df1.isna().sum().sum()
tot_vals = df1.shape[0] * df1.shape[1]
na_percentage = na_vals / tot_vals
print(f'{(na_percentage * 100):.2f}% of cells are missing values.')

