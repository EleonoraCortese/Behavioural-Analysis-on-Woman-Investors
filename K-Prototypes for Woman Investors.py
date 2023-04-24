#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install required packages
get_ipython().system('pip install kneed')
get_ipython().system('pip install kmodes')


# In[4]:


pip install yellowbrick


# In[1]:


# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from kmodes.kprototypes import KPrototypes

# See #1137: this allows compatibility for scikit-learn >= 0.24

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


# In[3]:


# load the dataset
df = pd.read_csv('path_name', index_col = 0) # devo cambiare la directory per caricarlo <3


# In[4]:


# view the dataset
df.head()


# In[94]:


# define a function that categorize the possession of investment products based on their risk.
# RAMO_III, MULTIRAMO and FONDI are related to relatively high risk so they are associated to 1.
#The other low risk product are associated to 0.

def categorise(row):  
    if row['FL_RAMO_III'] == 1 or row['FL_MULTIRAMO'] == 1 or row['FL_FONDI'] == 1:
        return 1
    else:
        return 0
    


# In[95]:


# create a new column containing 0 if the investor do not has risky product and 1 if she has
df['FL_RISCHIO'] = df.apply(lambda row: categorise(row), axis=1)


# In[96]:


# count the number of values in each category
df['FL_RISCHIO'].value_counts()


# In[97]:


# calculate the percentage of investments with respect to the total assets
def percinv(row):  
    if row["INVESTIMENTO"]>0:
        return row["INVESTIMENTO"]*100/row['PATRIMONIO']
    else:
        return 0
    


# In[98]:


# create a new column with the calculated values
df["%INV"] = df.apply(lambda row: percinv(row), axis=1)


# In[99]:


# create a new column calculated as the total amount allocated on high risk investment products 
df['PATRIMONIO_RISCHIO'] = df["PATRIMONIO_RAMO_III"]+df["PATRIMONIO_FONDI"]+df["PATRIMONIO_MULTIRAMO"]


# In[100]:


# calculate the percentage of risk-related assets with respect to the total investments
def percrischio(row):  
    if row["PATRIMONIO_RISCHIO"]>0:
        return row["PATRIMONIO_RISCHIO"]*100/row['INVESTIMENTO']
    else:
        return 0
    


# In[101]:


# create a new column with the calculated values 
df['%RISCHIO']= df.apply(lambda row: percrischio(row), axis=1)


# In[104]:


# create a new dataframe containing investors only
investors=df[df['FL_INVESTIMENTO']==1]
investors.drop('FL_INVESTIMENTO',1,inplace=True)
# create a new dataframe containing female investors only
finvestors=investors[investors['SESSO_M']==0]
finvestors.drop('SESSO_M',1,inplace=True)


# In[106]:


# import the required libraries
import numpy as np                                # For data management
import pandas as pd                               # For data management
import seaborn as sns                             # For data visualization and specifically for pairplot()
import matplotlib.pyplot as plt                   # For data visualization
from sklearn import datasets                      # To import the sample dataset
from sklearn.preprocessing import StandardScaler  # To transform the dataset
from sklearn.cluster import KMeans                # To instantiate, train and use model
from sklearn import metrics   


# In[17]:


# create a correlation matrix for variables in DataFrame
plt.figure(figsize = (70,40))
corr = finvestors.corr()
matrix = np.triu(corr)
sns.heatmap(corr, annot=True, mask = matrix)


# In[107]:


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


# In[108]:


# drop the most correlated columns + not useful columns for the cluster analysis
finvestors.drop(to_drop,1,inplace=True)


# In[111]:


# show the remaining columns
finvestors.columns


# In[112]:


# create a new df containing only the numeric variable PATRIMONIO 
# on which it's necessary to perform some transformation in order to normalyze it's distribution 
dftrans=finvestors[["PATRIMONIO"]]


# In[114]:


# View the "PATRIMONIO" column's distribution
sns.pairplot(data=dftrans, diag_kind='kde')


# In[115]:


# Log-transform the 'PATRIMONIO' column of the DataFrame dftrans
dftrans.PATRIMONIO=np.log(1+(dftrans.PATRIMONIO-min(dftrans.PATRIMONIO)))


# In[116]:


# View the 'PATRIMONIO' column's distribution after the Log-Transformation
sns.pairplot(data=dftrans, diag_kind='kde')


# In[1]:


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


# In[118]:


# Melt the DataFrame results_rs_pt and create a new DataFrame dfftrans_rs_pt_melted
dfftrans_rs_pt_melted=pd.melt(results_rs_pt)

# Create a box plot of the melted DataFrame dfftrans_rs_pt_melted
fig, ax = plt.subplots(figsize=(50,20))
ax=sns.boxplot(x='variable',y='value',data=dfftrans_rs_pt_melted)


# In[119]:


# View the 'PATRIMONIO' column's distribution after transformations
sns.pairplot(data=results_rs_pt, diag_kind='kde')


# In[120]:


# Drop the 'PATRIMONIO' column from the DataFrame finvestors
finvestors.drop("PATRIMONIO",1,inplace=True)
# Reset the index of the DataFrame finvestors
finvestors.reset_index(drop=True,inplace=True)
# Reset the index of the DataFrame results_rs_pt
results_rs_pt.reset_index(drop=True,inplace=True)
# Join the DataFrame finvestors with the DataFrame results_rs_pt and create a new DataFrame cluster
cluster=results_rs_pt.join(finvestors)


# ### K PROTOTYPES

# In[121]:


# Create a copy of the Dataframe cluster
prot=cluster.copy()
# Show the count of non-null values and the Dtype for each colum in the DataFrame
prot.info()


# In[122]:


# To perform K-Prototypes all the categorical variables have to be casted into "object" type so the following code do this step
prot["CLASSI_ETA"] = prot["CLASSI_ETA"].astype(object)
prot["FL_MULTIBANCA"] = prot["FL_MULTIBANCA"].astype(object)
prot["CLUSTER_DIGITAL"] = prot["CLUSTER_DIGITAL"].astype(object)
prot["TIPO_INTESTAZIONE_Monointestati"] = prot["TIPO_INTESTAZIONE_Monointestati"].astype(object)
prot["PROPENSIONE_AL_RISCHIO"] = prot["PROPENSIONE_AL_RISCHIO"].astype(object)
prot["FL_RISCHIO"] = prot["FL_RISCHIO"].astype(object)


# In[123]:


# Check the Dtype
prot.info()


# In[124]:


# get indices of category columns
catcol_idx = [prot.columns.get_loc(col) for col in list(prot.select_dtypes("object").columns)]
catcol_idx


# In[38]:


# Initialize a k-prototypes clustering model with 4 clusters
kproto4 = KPrototypes(n_clusters = 4, init='Cao', n_jobs = 4)
# Fit the clustering model to the data and predict cluster membership for each sample
# The `categorical` parameter specifies the indices of categorical features
res4 = kproto4.fit_predict(prot, categorical=catcol_idx)
# Calculate the cost (sum of distances) of the fitted clustering model
kproto4.cost_


# In[116]:


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


# In[207]:


# cost (sum distance): confirm visual clue of elbow plot
# KneeLocator class will detect elbows if curve is convex; if concave, will detect knees
cost_knee = KneeLocator(
        x=list(dict_cost.keys()), 
        y=list(dict_cost.values()), 
        S=0.1, curve="convex", direction="decreasing", online=True)

K_cost = cost_knee.elbow   
print("elbow at k =", f'{K_cost:.0f} clusters')


# In[40]:


# insert the column "Cluster" in the cluster Dataframe
cluster.insert(0,"Cluster",res4)


# In[43]:


# Display the Count of non-null values and the Dtype for each variable contained in the cluster DataFrame
cluster.info()


# In[44]:


# Create a barplot showing the number of female investors in each cluster
ax = cluster['Cluster'].value_counts().plot(kind='bar', title ="DISTRIBUZIONE cluster", figsize=(15, 10), fontsize=12)
ax.set_xlabel("CLUSTER", fontsize=12)
ax.set_ylabel("COUNT", fontsize=12)


# In[45]:


# count the number of female investors for each cluster
cluster['Cluster'].value_counts()


# In[46]:


# setting to display all columns in the DF
pd.set_option('max_columns', None)
# create a new Dataframe containing descriptive statistics for each variable grouped by clusters
descriptives=cluster.groupby('Cluster').describe()


# In[47]:


# display the results
descriptives


# In[48]:


#create an empty dataframe
summary=pd.DataFrame()


# In[49]:


#fill the empty dataframe with the correct measure of central tendency based on the Datatype 
summary["PATRIMONIO"]=(cluster.groupby('Cluster')["PATRIMONIO"].mean())
summary["CLASSI_ETA"]=(cluster.groupby('Cluster')["CLASSI_ETA"].median())
summary["FL_MULTIBANCA"]=(cluster.groupby('Cluster')["FL_MULTIBANCA"].agg(pd.Series.mode))
summary["CLUSTER_DIGITAL"]=(cluster.groupby('Cluster')["CLUSTER_DIGITAL"].median())
summary["PROPENSIONE_AL_RISCHIO"]=(cluster.groupby('Cluster')["PROPENSIONE_AL_RISCHIO"].median())
summary["TIPO_INTESTAZIONE_Monointestati"]=(cluster.groupby('Cluster')["TIPO_INTESTAZIONE_Monointestati"].agg(pd.Series.mode))
summary["FL_RISCHIO"]=(cluster.groupby('Cluster')["FL_RISCHIO"].agg(pd.Series.mode))


# In[51]:


ax = cluster.groupby(['Cluster']).CLUSTER_DIGITAL.value_counts().reset_index(name='counts').pivot(index='Cluster', columns='CLUSTER_DIGITAL', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("Cluster", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Cluster Digital vs Cluster', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 10.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[52]:


ax = cluster.groupby(['Cluster']).CLASSI_ETA.value_counts().reset_index(name='counts').pivot(index='Cluster', columns='CLASSI_ETA', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("Cluster", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Classi età vs Cluster', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 10.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[53]:


ax = cluster.groupby(['Cluster']).FL_MULTIBANCA.value_counts().reset_index(name='counts').pivot(index='Cluster', columns='FL_MULTIBANCA', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("Cluster", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Multibanca vs Cluster', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 10.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[132]:


ax = cluster.groupby(['Cluster']).FL_RISCHIO.value_counts().reset_index(name='counts').pivot(index='Cluster', columns='FL_RISCHIO', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("Cluster", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('FL_Rischio vs Cluster', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 10.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[133]:


ax = cluster.groupby(['Cluster']).PROPENSIONE_AL_RISCHIO.value_counts().reset_index(name='counts').pivot(index='Cluster', columns='PROPENSIONE_AL_RISCHIO', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("Cluster", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Propensione al rischio vs Cluster', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 10.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[256]:



ax = cluster.groupby(['Cluster']).TIPO_INTESTAZIONE_Monointestati.value_counts().reset_index(name='counts').pivot(index='Cluster', columns='TIPO_INTESTAZIONE_Monointestati', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("Cluster", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Tipo Intestazione vs Cluster', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 10.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[213]:


cluster.groupby(['Cluster']).PATRIMONIO.mean()


# In[214]:


#percentuali di patrimonio allocate sui pdt
patrimonio=[-1.215,-0.060,0.160,1.350]
cluster=['0','1','2','3']
figure2 = plt.bar(cluster, patrimonio)
plt.legend(fontsize=25)
plt.title("PATRIMONIO DISPONIBILE",fontsize=15)
plt.xlabel('PATRIMONIO',fontsize=15)
plt.ylabel('€',fontsize=15)
plt.rcParams["figure.figsize"] = [25.00, 11.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[313]:


cl=cluster[["Cluster"]]
cl.reset_index(drop=True,inplace=True)


# In[314]:


finvestors.reset_index(drop=True,inplace=True)


# In[315]:


dfcluster=cl.join(finvestors)


# In[316]:


dfcluster.info()


# In[317]:


dfcluster.to_csv('Cluster_Investitrici_Completo.csv')


# In[318]:


cl0=dfcluster[dfcluster["Cluster"]==0]
cl1=dfcluster[dfcluster["Cluster"]==1]
cl2=dfcluster[dfcluster["Cluster"]==2]
cl3=dfcluster[dfcluster["Cluster"]==3]


# In[356]:


print('CLUSTER 1: ETA MEDIA NON ELEVATISSIMA, PROPENSIONE AL RISCHIO MEDIA, ABBASTANZA RISCHIO, PATRIMONIO MEDIO-ALTO')
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


# In[358]:


print('CLUSTER 2: ANZIANE, INVESTONO MOLTO MA POCO IN RISCHIO, PROPENSIONE RISCHIO BASSA, PATRIMONIO ABBASTANZA ELEVATO')
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


# In[359]:


print('CLUSTER 3: RICCHE, INVESTITRICI, RISCHIO, ETA MEDIA ABBASTANZA ELEVATA')
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


# In[360]:


print('CLUSTER 0: MENO RICCHE, ETA MEDIA PIU BASSA, NO RISCHIO')
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


# In[331]:


cl0.columns


# In[125]:


#calcolo indice di shilouette

from kmodes.kmodes import KModes
from kmodes import kprototypes
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
def mixed_distance(a,b,categorical=None, alpha=0.01):
    if categorical is None:
        num_score=kprototypes.euclidean_dissim(a,b)
        return num_score
    else:
        cat_index=categorical
        a_cat=[]
        b_cat=[]
        for index in cat_index:
            a_cat.append(a[index])
            b_cat.append(b[index])
        a_num=[]
        b_num=[]
        l=len(a)
        for index in range(l):
            if index not in cat_index:
                a_num.append(a[index])
                b_num.append(b[index])
                
        a_cat=np.array(a_cat).reshape(1,-1)
        a_num=np.array(a_num).reshape(1,-1)
        b_cat=np.array(b_cat).reshape(1,-1)
        b_num=np.array(b_num).reshape(1,-1)
        cat_score=kprototypes.matching_dissim(a_cat,b_cat)
        num_score=kprototypes.euclidean_dissim(a_num,b_num)
        return cat_score+num_score*alpha
def dm_prototypes(dataset,categorical=None,alpha=0.1):
    #if the input dataset is a dataframe, we take out the values as a numpy. 
    #If the input dataset is a numpy array, we use it as is.
    if type(dataset).__name__=='DataFrame':
        dataset=dataset.values    
    lenDataset=len(dataset)
    distance_matrix=np.zeros(lenDataset*lenDataset).reshape(lenDataset,lenDataset)
    for i in range(lenDataset):
        for j in range(lenDataset):
            x1= dataset[i]
            x2= dataset[j]
            distance=mixed_distance(x1, x2,categorical=categorical,alpha=alpha)
            distance_matrix[i][j]=distance
            distance_matrix[j][i]=distance
    return distance_matrix



# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

#obtain array of values
data_array=prot.values
#dizionario vuoto per storare l'indice
silhouette_scores = dict()
K = range(2,10)

distance_matrix=dm_prototypes(data_array,categorical=catcol_idx,alpha=0.1)
for k in K:
    untrained_model = kprototypes.KPrototypes(n_clusters=k,max_iter=20)
    trained_model = untrained_model.fit(data_array, categorical=catcol_idx)
    cluster_labels = trained_model.labels_
    score=silhouette_score(distance_matrix, cluster_labels,metric="precomputed")
    silhouette_scores[k]=score
print("The k and associated Silhouette scores are: ",silhouette_scores)


# In[ ]:




