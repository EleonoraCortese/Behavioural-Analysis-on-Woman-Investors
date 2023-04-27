# install required packages
pip install kneed
pip install kmodes
pip install yellowbrick

# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes

#this allows compatibility for scikit-learn >= 0.24

from sklearn.utils import _safe_indexing
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from kneed import KneeLocator
from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
import warnings

# ignore warning messages
warnings.filterwarnings("ignore")

# set parameters
frTRAIN = 0.8               # % size of training dataset
RNDN = 42                   # random state
nK = 12                     # initial guess: clusters

# set options for displaying floating point numbers
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.precision",2)
np.set_printoptions(precision=3, suppress=True)
pd.options.display.float_format = '{:.3f}'.format

# load the dataset
df = pd.read_csv('path_name', index_col = 0) # devo cambiare la directory per caricarlo <3

# view the dataset
df.head()

# define a function that categorize the possession of investment products based on their risk.
# RAMO_III, MULTIRAMO and FONDI are related to relatively high risk so they are associated to 1.
#The other low risk product are associated to 0.

def categorise(row):  
    if row['FL_RAMO_III'] == 1 or row['FL_MULTIRAMO'] == 1 or row['FL_FONDI'] == 1:
        return 1
    else:
        return 0

# create a new column containing 0 if the investor do not has risky product and 1 if she has
df['FL_RISCHIO'] = df.apply(lambda row: categorise(row), axis=1)

# count the number of values in each category
df['FL_RISCHIO'].value_counts()

# calculate the percentage of investments with respect to the total assets
def percinv(row):  
    if row["INVESTIMENTO"]>0:
        return row["INVESTIMENTO"]*100/row['PATRIMONIO']
    else:
        return 0

# create a new column with the calculated values
df["%INV"] = df.apply(lambda row: percinv(row), axis=1)

# create a new column calculated as the total amount allocated on high risk investment products 
df['PATRIMONIO_RISCHIO'] = df["PATRIMONIO_RAMO_III"]+df["PATRIMONIO_FONDI"]+df["PATRIMONIO_MULTIRAMO"]

# calculate the percentage of risk-related assets with respect to the total investments
def percrischio(row):  
    if row["PATRIMONIO_RISCHIO"]>0:
        return row["PATRIMONIO_RISCHIO"]*100/row['INVESTIMENTO']
    else:
        return 0
    
# create a new column with the calculated values 
df['%RISCHIO']= df.apply(lambda row: percrischio(row), axis=1)

# create a new dataframe containing investors only
investors=df[df['FL_INVESTIMENTO']==1]
investors.drop('FL_INVESTIMENTO',1,inplace=True)
# create a new dataframe containing female investors only
finvestors=investors[investors['SESSO_M']==0]
finvestors.drop('SESSO_M',1,inplace=True)

# import the required libraries
import numpy as np                                # For data management
import pandas as pd                               # For data management
import seaborn as sns                             # For data visualization and specifically for pairplot()
import matplotlib.pyplot as plt                   # For data visualization
from sklearn import datasets                      # To import the sample dataset
from sklearn.preprocessing import StandardScaler  # To transform the dataset
from sklearn.cluster import KMeans                # To instantiate, train and use model
from sklearn import metrics   

# create a correlation matrix for variables in DataFrame
plt.figure(figsize = (70,40))
corr = finvestors.corr()
matrix = np.triu(corr)
sns.heatmap(corr, annot=True, mask = matrix)

#create a list containing the most correlated columns
to_drop=['%INV','%RISCHIO','FL_ACCR_STIP_PENS_CC','FL_BFP','FL_CCBP','FL_PREVIDENZA',
         'FL_POSSESSO_EMAIL','PATRIMONIO_PERSONA','PATRIMONIO_RAMO_III','PATRIMONIO_RAMO_IV',
         'PATRIMONIO_MULTIRAMO','SEGMENTO','PATRIMONIO_BFP','PATRIMONIO_RAMO_I','RISPARMIO',
         'INVESTIMENTO','PATRIMONIO_CONTI','PATRIMONIO_LIBRETTI','DS_CLUSTER_DIGITAL','FL_MIFID_SCADUTO',
         'DS_PORTAFOGLIO_PERSONAL','AREA_TERRITORIALE_MA SUD','AREA_TERRITORIALE_MA SICILIA',
         'AREA_TERRITORIALE_MA NORD OVEST','FL_FONDI','FL_MULTIRAMO','AREA_TERRITORIALE_MA NORD EST',
         'AREA_TERRITORIALE_MA CENTRO NORD',"FL_RAMO_I",'FL_MIFID_MAI_PROFILATO','FL_LIB_SMART', 'FL_LIB_SUPER_SMART',
         'FL_LIB', 'FL_TITOLI', 'FL_PERSONA', 'FL_RAMO_III', 'FL_RAMO_IV',
         'FL_POSTEPAY', 'FL_EVOLUTION', 'PATRIMONIO_TITOLI', 'PATRIMONIO_FONDI',
         'PATRIMONIO_PREVIDENZA', 'PATRIMONIO_POSTEPAY', 'PATRIMONIO_EVOLUTION',
         'FL_LIQUIDITA', 'FL_RISPARMIO', 'LIQUIDITA',
         'FL_APP_BP', 'FL_APP_PPAY', 'FL_MIFID_PROFILATO',
         'DS_PORTAFOGLIO_PREMIUM','DS_PORTAFOGLIO_VENDITORI MOBILI', 'REGIONE_Basilicata',
         'REGIONE_Calabria', 'REGIONE_Campania', 'REGIONE_Emilia Romagna',
         'REGIONE_Friuli Venezia Giulia', 'REGIONE_Lazio', 'REGIONE_Liguria',
         'REGIONE_Lombardia', 'REGIONE_Marche', 'REGIONE_Molise',
         'REGIONE_Piemonte', 'REGIONE_Puglia', 'REGIONE_Sardegna',
         'REGIONE_Sicilia', 'REGIONE_Toscana', 'REGIONE_Trentino Alto Adige',
         'REGIONE_Umbria', "REGIONE_Val D'Aosta", "REGIONE_Veneto","PATRIMONIO_RISCHIO"]

# drop the most correlated columns + not useful columns for the cluster analysis
finvestors.drop(to_drop,1,inplace=True)

# show the remaining columns
finvestors.columns

# create a new df containing only the numeric variable PATRIMONIO 
# on which it's necessary to perform some transformation in order to normalyze it's distribution 
dftrans=finvestors[["PATRIMONIO"]]

# View the "PATRIMONIO" column's distribution
sns.pairplot(data=dftrans, diag_kind='kde')

# Log-transform the 'PATRIMONIO' column of the DataFrame dftrans
dftrans.PATRIMONIO=np.log(1+(dftrans.PATRIMONIO-min(dftrans.PATRIMONIO)))

# View the 'PATRIMONIO' column's distribution after the Log-Transformation
sns.pairplot(data=dftrans, diag_kind='kde')

#Import the required libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

# Create a pipeline with RobustScaler and PowerTransformer transformers
trans_rs_pt = Pipeline(steps=[("rs",RobustScaler()),("pt", PowerTransformer())])

# Fit the pipeline with the DataFrame dftrans
trans_rs_pt = trans_rs_pt.fit(dftrans)

# Transform the DataFrame dftrans using the pipeline and create a new DataFrame results_rs_pt
results_rs_pt= pd.DataFrame(trans_rs_pt.transform(dftrans), columns=list(dftrans))
results_rs_pt.head()

# Melt the DataFrame results_rs_pt and create a new DataFrame dfftrans_rs_pt_melted
dfftrans_rs_pt_melted=pd.melt(results_rs_pt)

# Create a box plot of the melted DataFrame dfftrans_rs_pt_melted
fig, ax = plt.subplots(figsize=(50,20))
ax=sns.boxplot(x='variable',y='value',data=dfftrans_rs_pt_melted)

# View the 'PATRIMONIO' column's distribution after transformations
sns.pairplot(data=results_rs_pt, diag_kind='kde')

# Drop the 'PATRIMONIO' column from the DataFrame finvestors
finvestors.drop("PATRIMONIO",1,inplace=True)

# Reset the index of the DataFrame finvestors
finvestors.reset_index(drop=True,inplace=True)

# Reset the index of the DataFrame results_rs_pt
results_rs_pt.reset_index(drop=True,inplace=True)

# Join the DataFrame finvestors with the DataFrame results_rs_pt and create a new DataFrame cluster
cluster=results_rs_pt.join(finvestors)


## K PROTOTYPES

# Create a copy of the Dataframe cluster
prot=cluster.copy()

# Show the count of non-null values and the Dtype for each colum in the DataFrame
prot.info()


# To perform K-Prototypes all the categorical variables have to be casted into "object" type so the following code does this step
prot["CLASSI_ETA"] = prot["CLASSI_ETA"].astype(object)
prot["FL_MULTIBANCA"] = prot["FL_MULTIBANCA"].astype(object)
prot["CLUSTER_DIGITAL"] = prot["CLUSTER_DIGITAL"].astype(object)
prot["TIPO_INTESTAZIONE_Monointestati"] = prot["TIPO_INTESTAZIONE_Monointestati"].astype(object)
prot["PROPENSIONE_AL_RISCHIO"] = prot["PROPENSIONE_AL_RISCHIO"].astype(object)
prot["FL_RISCHIO"] = prot["FL_RISCHIO"].astype(object)

# Check the Dtype
prot.info()

# get indices of category columns
catcol_idx = [prot.columns.get_loc(col) for col in list(prot.select_dtypes("object").columns)]
catcol_idx

# Initialize a k-prototypes clustering model with 4 clusters
kproto4 = KPrototypes(n_clusters = 4, init='Cao', n_jobs = 4)

# Fit the clustering model to the data and predict cluster membership for each sample
# The `categorical` parameter specifies the indices of categorical features
res4 = kproto4.fit_predict(prot, categorical=catcol_idx)

# Calculate the cost (sum of distances) of the fitted clustering model
kproto4.cost_

# kprototypes: looking for the elbow - compare number of clusters by their cost
# cost = sum distance of all points to their respective cluster centroids
# run kPrototypes for alternative number of clusters k
dict_cost = {}
for k in tqdm(range(2,9)):
    kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=4, verbose=2)
    res = kproto.fit_predict(prot, categorical=catcol_idx)
    dict_cost[k] = kproto.cost_
print("cost (sum distance) for all k:")
_ = [print(k,":",f'{v:.3f}') for k,v in dict_cost.items()]

# scree plot: look for elbow
plt.figure(figsize=[8,5])
plt.plot(dict_cost.keys(), dict_cost.values(), color="blue")
plt.title("cost (sum distance) vs. number of clusters")
plt.xticks(np.arange(2,10,1))
plt.xlabel("number of clusters K")
plt.ylabel("cost");

# cost (sum distance): confirm visual clue of elbow plot
# KneeLocator class will detect elbows if curve is convex; if concave, will detect knees
cost_knee = KneeLocator(
        x=list(dict_cost.keys()), 
        y=list(dict_cost.values()), 
        S=0.1, curve="convex", direction="decreasing", online=True)
K_cost = cost_knee.elbow   
print("elbow at k =", f'{K_cost:.0f} clusters')

# insert the column "Cluster" in the cluster Dataframe
cluster.insert(0,"Cluster",res4)

# Display the Count of non-null values and the Dtype for each variable contained in the cluster DataFrame
cluster.info()

# Create a barplot showing the number of female investors in each cluster
ax = cluster['Cluster'].value_counts().plot(kind='bar', title ="DISTRIBUZIONE cluster", figsize=(15, 10), fontsize=12)
ax.set_xlabel("CLUSTER", fontsize=12)
ax.set_ylabel("COUNT", fontsize=12)

# count the number of female investors for each cluster
cluster['Cluster'].value_counts()

# setting to display all columns in the DF
pd.set_option('max_columns', None)

# create a new Dataframe containing descriptive statistics for each variable grouped by clusters
descriptives=cluster.groupby('Cluster').describe()

# display the results
descriptives

#create an empty dataframe
summary=pd.DataFrame()

#fill the empty dataframe with the correct measure of central tendency based on the Datatype 
summary["PATRIMONIO"]=(cluster.groupby('Cluster')["PATRIMONIO"].mean())
summary["CLASSI_ETA"]=(cluster.groupby('Cluster')["CLASSI_ETA"].median())
summary["FL_MULTIBANCA"]=(cluster.groupby('Cluster')["FL_MULTIBANCA"].agg(pd.Series.mode))
summary["CLUSTER_DIGITAL"]=(cluster.groupby('Cluster')["CLUSTER_DIGITAL"].median())
summary["PROPENSIONE_AL_RISCHIO"]=(cluster.groupby('Cluster')["PROPENSIONE_AL_RISCHIO"].median())
summary["TIPO_INTESTAZIONE_Monointestati"]=(cluster.groupby('Cluster')["TIPO_INTESTAZIONE_Monointestati"].agg(pd.Series.mode))
summary["FL_RISCHIO"]=(cluster.groupby('Cluster')["FL_RISCHIO"].agg(pd.Series.mode))

# Create a new DataFrame containing the column "Cluster" Only
cl=cluster[["Cluster"]]
# Reset the index of the cl DataFrame
cl.reset_index(drop=True,inplace=True)
# Reset the index of the finvestors Dataframe
finvestors.reset_index(drop=True,inplace=True)
# Join cl and finvestors in order to have the complete Dataset containing all the variables
dfcluster=cl.join(finvestors)

#Save the DataFrame as a csv file
dfcluster.to_csv('Cluster_Investitrici_Completo.csv')


## PERFORMING SOME BASIC STATISTICS TO DESCRIBE EACH CLUSTER

# Create 4 Dataframe based on the cluster membership
cl0=dfcluster[dfcluster["Cluster"]==0]
cl1=dfcluster[dfcluster["Cluster"]==1]
cl2=dfcluster[dfcluster["Cluster"]==2]
cl3=dfcluster[dfcluster["Cluster"]==3]

# Display some statistics to describe the CLUSTER 0
print('CLUSTER 0:')
print(f'Patrimonio medio:{round(cl0.PATRIMONIO.mean(),2)}€')
print(f'Investimento medio:{round(cl0.INVESTIMENTO.mean(),2)}€')
print(f'%Rischio medio:{round(cl0["%RISCHIO"].mean(),2)}%')
print(f'Risparmio medio:{round(cl0["RISPARMIO"].mean(),2)}€')
print(f'Liquidità medio:{round(cl0["LIQUIDITA"].mean(),2)}€')
print(f'Patrimonio rischio medio:{round(cl0["PATRIMONIO_RISCHIO"].mean(),2)}€')
print(f'Distribuzione propensione al rischio:\n{cl0.PROPENSIONE_AL_RISCHIO.value_counts()}')
print(f'Distribuzione classi di età:\n{cl0.CLASSI_ETA.value_counts()}')
print(f'Segmento:\n{cl0.SEGMENTO.value_counts()}')
print(f'%Investimento medio:{round(cl0["%INV"].mean(),2)}%')

# Display some statistics to describe the CLUSTER 1
print('CLUSTER 1:')
print(f'Patrimonio medio:{round(cl1.PATRIMONIO.mean(),2)}€')
print(f'Investimento medio:{round(cl1.INVESTIMENTO.mean(),2)}€')
print(f'%Rischio medio:{round(cl1["%RISCHIO"].mean(),2)}%')
print(f'Risparmio medio:{round(cl1["RISPARMIO"].mean(),2)}€')
print(f'Liquidità medio:{round(cl1["LIQUIDITA"].mean(),2)}€')
print(f'Patrimonio rischio medio:{round(cl1["PATRIMONIO_RISCHIO"].mean(),2)}€')
print(f'Distribuzione propensione al rischio:\n{cl1.PROPENSIONE_AL_RISCHIO.value_counts()}')
print(f'Distribuzione classi di età:\n{cl1.CLASSI_ETA.value_counts()}')
print(f'%Investimento medio:{round(cl1["%INV"].mean(),2)}%')
print(f'Segmento:\n{cl1.SEGMENTO.value_counts()}')

# Display some statistics to describe the CLUSTER 2
print('CLUSTER 2:')
print(f'Patrimonio medio:{round(cl2.PATRIMONIO.mean(),2)}€')
print(f'Investimento medio:{round(cl2.INVESTIMENTO.mean(),2)}€')
print(f'%Rischio medio:{round(cl2["%RISCHIO"].mean(),2)}%')
print(f'Risparmio medio:{round(cl2["RISPARMIO"].mean(),2)}€')
print(f'Liquidità medio:{round(cl2["LIQUIDITA"].mean(),2)}€')
print(f'Patrimonio rischio medio:{round(cl2["PATRIMONIO_RISCHIO"].mean(),2)}€')
print(f'Distribuzione propensione al rischio:\n{cl2.PROPENSIONE_AL_RISCHIO.value_counts()}')
print(f'Distribuzione classi di età:\n{cl2.CLASSI_ETA.value_counts()}')
print(f'%Investimento medio:{round(cl2["%INV"].mean(),2)}%')
print(f'Segmento:\n{cl2.SEGMENTO.value_counts()}')

# Display some statistics to describe the CLUSTER 3
print('CLUSTER 3:')
print(f'Patrimonio medio:{round(cl3.PATRIMONIO.mean(),2)}€')
print(f'Investimento medio:{round(cl3.INVESTIMENTO.mean(),2)}€')
print(f'%Rischio medio:{round(cl3["%RISCHIO"].mean(),2)}%')
print(f'Risparmio medio:{round(cl3["RISPARMIO"].mean(),2)}€')
print(f'Liquidità medio:{round(cl3["LIQUIDITA"].mean(),2)}€')
print(f'Patrimonio rischio medio:{round(cl3["PATRIMONIO_RISCHIO"].mean(),2)}€')
print(f'Distribuzione propensione al rischio:\n{cl3.PROPENSIONE_AL_RISCHIO.value_counts()}')
print(f'Distribuzione classi di età:\n{cl3.CLASSI_ETA.value_counts()}')
print(f'Segmento:\n{cl3.SEGMENTO.value_counts()}')
print(f'%Investimento medio:{round(cl3["%INV"].mean(),2)}%')

