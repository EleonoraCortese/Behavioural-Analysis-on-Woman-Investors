#The following code performs some cleaning and imputing procedures required to conduct more complex quatitative data manipulations

#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from kneed import KneeLocator
from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

#read the dataset
df = pd.read_csv('path')

#view the dataset
df.head()

#check variables
df.columns


## DATA CLEANING E IMPUTING

#drop unnecessary columns
df.drop(['Unnamed: 0','ID_CLIE','FL_SOTT_BFP',
       'FL_SOTT_FONDI', 'FL_SOTT_MULTIRAMO', 'FL_SOTT_PREVIDENZA',
       'FL_SOTT_PROTEZIONE', 'FL_SOTT_RAMO_I', 'FL_SOTT_SUPERSMART',
       'FL_SOTT_TITOLI', 'SOTT_BFP_IMP', 'SOTT_FONDI_IMP',
       'SOTT_MULTIRAMO_IMP', 'SOTT_PREVIDENZA_IMP', 'SOTT_PROTEZIONE_IMP',
       'SOTT_RAMO_I_IMP', 'SOTT_SUPERSMART_IMP', 'SOTT_TITOLI_IMP',
       'FL_RACCOLTA_2022'], inplace=True, axis=1)

#check for missing values
print(df.info()) 
print(df.shape[0])
na_vals = df.isna().sum().sum()
tot_vals = df.shape[0] * df.shape[1]
na_percentage = na_vals / tot_vals
print(f'{(na_percentage * 100):.2f}% of cells are missing values.')

#NaN values come from the "CLUSTER_DIGITAL" and "DS_PORTAFOGLIO" columns, 
#however, by analyzing the "DS_CLUSTER_DIGITAL" column, we can see that the "Not applicable" mode can be attributed to a NaN
df.DS_CLUSTER_DIGITAL.unique()

#converting variables from categorical to numerical with "df.replace()".
replace_map = {'CLASSI_ETA': {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65-74': 6, '75+': 7}}
df.replace(replace_map, inplace=True)
print(df['CLASSI_ETA'].unique())
replace_map1 = {'CLUSTER_DIGITAL': {'D0': 0, 'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4}}
df.replace(replace_map1, inplace=True)
print(df['CLUSTER_DIGITAL'].unique())
replace_map2 = {'DS_CLUSTER_DIGITAL': {'Digital dispositivi evoluti, sottoscrittori di prodotti': 5,
 'Digital dispositivi basici transazionali':4, 'Digital informativi':3,
 'Digital dispositivi dormienti':2, 'No Registrati poste.it':1, 'No Digital':0}}
df.replace(replace_map2, inplace=True)
#replacing"Non applicabile" with nan
df['DS_CLUSTER_DIGITAL'].replace('Non applicabile', np.nan, inplace=True)
print(df['DS_CLUSTER_DIGITAL'].unique())
replace_map3 = {'SEGMENTO': {'MASS':1,'AFFLUENT':2, 'PRIVATE':3, 'PREMIUM':4}}
df.replace(replace_map3, inplace=True)
print(df['SEGMENTO'].unique())

#calculating the percentage of missing values in the sample
print(df.shape[0])
na_vals = df.isna().sum().sum()
tot_vals = df.shape[0] * df.shape[1]
na_percentage = na_vals / tot_vals
print(f'{(na_percentage * 100):.2f}% of cells are missing values.')

#checking for variables with missing values
print(df.info()) 

#Adopting a simple imputer that uses the mode to compute missing values in the DS_PORTAFOGLIO column before converting it back to dummy
#For the other two columns where there are NaN values, it's possible to use KNN since they have been converted to numerical values.

#using a simple imputer to impute the missing values based on the most frequent value for the DS_PORTAFOGLIO column
modecol = ['DS_PORTAFOGLIO']
dfmode=df[modecol]
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
imputed = pd.DataFrame(si.fit_transform(dfmode))
#re-name the variable in order to give it back its original name
imputed.rename(columns = {0:'DS_PORTAFOGLIO'}, inplace = True)
imputed.reset_index(drop=True,inplace=True)

#removing the old column from the dataset and resetting the index
df.drop(labels=['DS_PORTAFOGLIO'], axis=1, inplace=True)
df.reset_index(drop=True,inplace=True)

#inserting the new column in which the missing values are imputed
df1 = pd.merge(df, imputed, left_index=True, right_index=True)

#converting not ordinal categorical variables into dummy using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="first")
ohe_df = pd.DataFrame(ohe.fit_transform(df1[['SESSO', 'DS_PORTAFOGLIO', 'AREA_TERRITORIALE', 'REGIONE',
                                             'TIPO_INTESTAZIONE']]).toarray(), columns = ohe.get_feature_names_out())
ohe_df = ohe_df.astype('int64')

#drop old columns containing not ordinal categorical variables
df1.drop(labels=['SESSO', 'DS_PORTAFOGLIO', 'AREA_TERRITORIALE', 'REGIONE',
                 'TIPO_INTESTAZIONE'], axis=1, inplace=True)

#reset the index
df1.reset_index(drop=True,inplace=True)
ohe_df.reset_index(drop=True,inplace=True)

#inserting the new columns containing the dummies in the original dataframe
df2 = pd.merge(df1, ohe_df, left_index=True, right_index=True)
df2.head()

#using a KNN imputer to compute missing values for CLUSTER_DIGITAL and DS_CLUSTER_DIGITAL
from sklearn.impute import KNNImputer
knn = KNNImputer()
df3=pd.DataFrame(knn.fit_transform(df2), columns=list(df2))

#checking for missing values (there aren't missing values anymore)
na_vals = df3.isna().sum().sum()
tot_vals = df3.shape[0] * df3.shape[1]
na_percentage = na_vals / tot_vals
print(f'{(na_percentage * 100):.2f}% of cells are missing values.')

#checking for the imputing procedure of the ordinal category 
ordinali=['CLUSTER_DIGITAL','DS_CLUSTER_DIGITAL',]
for i in ordinali:
    print(df3[i].unique())

#casting ordinals in int to fit in the closest category
df3 = df3.astype({"CLUSTER_DIGITAL":"int","DS_CLUSTER_DIGITAL":"int"})
for i in ordinali:
    print(df3[i].unique())
