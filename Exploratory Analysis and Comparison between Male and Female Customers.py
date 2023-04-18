# the code performs some basic exploratory data analysis and creates some visualizations to help understand the dataset

#Imports necessary Python libraries including NumPy, Pandas, Matplotlib, Seaborn, and SciPy.
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

#Reading the CSV file into a Pandas DataFrame object named df.
df = pd.read_csv('/Users/cortess/Desktop/Tesi/Clean4descriptive.csv')

#Displaying the first few rows of the DeteFrame
df.head()

#Displaying all the columns contained in the Dataframe 
print(df.columns)

#Dropping the unnecessary columns from the DataFrame.
df.drop(['Unnamed: 0'], inplace=True, axis=1)

#Printing the summary statistics of the DataFrame using the describe() method.
df.describe()

# Calculating and printing the percentage of male and female customers in the sample 
df1=df.copy()
print("Percentuale di uomini e donne nel campione")
perc=list(df1['SESSO'].value_counts()/200000*100)
print(f'donne:{round(perc[0],2)}%')
print(f'uomini:{round(perc[1],2)}%')

#Creating a pie chart to visualize the result.
labels = 'Donne', 'Uomini'
sizes = [perc[0], perc[1]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'fontsize': 15})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#Creating a bar plot to show the distribution of age groups in the sample.
classi = ['18-24','25-34','35-44','45-54','55-64','65-74','75+']
frequenze = [4405,13694,22346,35246,42697,40764,40848]
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("DISTRIBUZIONE PER CLASSI D'ETA'")
plt.ylabel('COUNT', fontsize=12)
plt.xlabel('CLASSI D\'ETA',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.bar(classi, frequenze)
plt.show()

#Calculating the total patrimony in the sample
df1['PATRIMONIO'].sum()


#Calculating and printing the average age in the sample
eta_count=list(df1.CLASSI_ETA.value_counts())
val_centr=[(55+64)/2,(65+74)/2,(75+90)/2,(45+54)/2,(35+44)/2,(25+34)/2,(18+24)/2]
pondera = list(zip(val_centr,eta_count))
ponderata=[]
for i in pondera:
    ponderata.append((i[0]*i[1])/df1.shape[0])
print(f'età media: {round(sum(ponderata),2)}')


#creating a bar plot to show the average patrimony for each age group.
ax1 = df1.groupby('CLASSI_ETA')['PATRIMONIO'].mean().plot(kind='bar', title ="PATRIMONIO MEDIO PER CLASSI D'ETÁ", figsize=(15, 10), legend=False, fontsize=12)
ax1.set_xlabel("CLASSI D'ETÁ", fontsize=12)
ax1.set_ylabel("PATRIMONIO IN € ", fontsize=12)

#Creating a bar plot to show the distribution of risk propensity in the sample.
propensione = ['1','2','3','4','5']
frequenze = [9539,94968,81223,13338,932]
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("DISTRIBUZIONE PER PROPENSIONE AL RISCHIO")
plt.ylabel('COUNT', fontsize=12)
plt.xlabel('PROPENSIONE AL RISCHIO',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.bar(propensione, frequenze)
plt.show()

#Creating a bar plot to show the distribution of customers for each segment
ax = df1['SEGMENTO'].value_counts().plot(kind='bar', title ="DISTRIBUZIONE PER SEGMENTO", figsize=(15, 10), fontsize=12)
ax.set_xlabel("SEGMENTO", fontsize=12)
ax.set_ylabel("COUNT", fontsize=12)

#Creating a bar plot to show the average patrimony for each segment 
ax1 = df1.groupby('SEGMENTO')['PATRIMONIO'].mean().plot(kind='bar', title ="Patrimonio medio vs segmento", figsize=(15, 10), legend=True, fontsize=12)
ax1.set_xlabel("Segmento", fontsize=12)
ax1.set_ylabel("Patrimonio", fontsize=12)

#Calculating and printing the average wealth for multibank and non-multibank customers 
mb=list(df1.FL_MULTIBANCA.value_counts()/200000*100)
print(f'Patrimonio medio vs multibanca:\n',round(df1.groupby('FL_MULTIBANCA')['PATRIMONIO'].mean(),3))

#creating a pie chart to visualize the result
labels1 = 'Multibanca', 'Non-multibanca'
sizes1 = [mb[0], mb[1]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'fontsize': 13})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#Creating a bar plot to show the average wealth for each gender
ax1 = df1.groupby('SESSO')['PATRIMONIO'].mean().plot(kind='bar', title ="Patrimonio medio vs sesso", figsize=(15, 10), legend=True, fontsize=12)
ax1.set_xlabel("Sesso", fontsize=12)
ax1.set_ylabel("Patrimonio", fontsize=12)

#Creating a bar plot to show the distribution of customers in different geographic areas
ax = df1['AREA_TERRITORIALE'].value_counts().plot(kind='bar', title ="DISTRIBUZIONE PER AREA TERRITORIALE", figsize=(15, 10), fontsize=12)
ax.set_xlabel("AREA TERRITORIALE", fontsize=12)
ax.set_ylabel("COUNT", fontsize=12)

#Printing the average "PATRIMONIO" (wealth) per "AREA_TERRITORIALE" (territorial area)
print(f'Media patrimonio per area territoriale:\n',round(df1.groupby('AREA_TERRITORIALE')['PATRIMONIO'].mean(),3))

#Creating a bar plot of the average "PATRIMONIO" (wealth) per "AREA_TERRITORIALE" (territorial area)
ax1 = df1.groupby('AREA_TERRITORIALE')['PATRIMONIO'].mean().plot(kind='bar', title ="Patrimonio medio per area territoriale", figsize=(15, 10), legend=True, fontsize=12)
ax1.set_xlabel("Area territoriale", fontsize=12)
ax1.set_ylabel("Patrimonio", fontsize=12)

# Print the percentage of customers per "REGIONE" (region)
print(df1.REGIONE.value_counts()/200000*100)

# Print the average "PATRIMONIO" (wealth) per "REGIONE" (region)
print(f'Media patrimonio per regione:\n',round(df1.groupby('REGIONE')['PATRIMONIO'].mean(),3))

# Create a bar plot of the number of customers per "REGIONE" (region)
ax = df1['REGIONE'].value_counts().plot(kind='bar', title ="DISTRIBUZIONE PER REGIONE", figsize=(15, 10), fontsize=12)
ax.set_xlabel("REGIONE", fontsize=12)
ax.set_ylabel("COUNT", fontsize=12)

# Print the percentage of customers per "CLUSTER_DIGITAL" (digital cluster)
print(df1.CLUSTER_DIGITAL.value_counts()/200000*100)

# Create a bar plot of the percentage of customers per "CLUSTER_DIGITAL" (digital cluster)
cluster=['D0','D1','D2','D3','D4']
perc=[44.2540,8.4745,19.3365,25.3845,2.5505]
figure2 = plt.bar(cluster, perc)
plt.title("DIGITALIZZAZIONE DELLA CLIENTELA",fontsize=15)
plt.rcParams["figure.figsize"] = [10.00, 11.50]
plt.xlabel('CLUSTER DIGITAL',fontsize=15 )
plt.ylabel('% SAMPLE',fontsize=15)
plt.show()

# Create a bar plot of the average "PATRIMONIO" (wealth) per "CLUSTER_DIGITAL" (digital cluster)
ax1 = df1.groupby('CLUSTER_DIGITAL')['PATRIMONIO'].mean().plot(kind='bar', title ="Patrimonio medio per cluster digital", figsize=(15, 10), legend=True, fontsize=12)
ax1.set_xlabel("CLUSTER_DIGITAL", fontsize=12)
ax1.set_ylabel("Patrimonio", fontsize=12)

# Create a grouped bar plot of the number of customers per "CLASSI_ETA" (age class) and "CLUSTER_DIGITAL" (digital cluster)
ax = df1.groupby(['CLASSI_ETA']).CLUSTER_DIGITAL.value_counts().reset_index(name='counts').pivot(index='CLASSI_ETA', columns='CLUSTER_DIGITAL', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("CLASSI D'ETÀ", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Cluster Digital vs Classi d\'età', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#printing the average wealth in the sample
print(f'il partimonio medio è di: {round(df1.PATRIMONIO.mean(),2)}€')

#Calculating the percentage of the three product categories: Liquidity, Savings and Investments on the overall wealth
L=df1['LIQUIDITA'].mean()*100/df1["PATRIMONIO"].mean()
R=df1['RISPARMIO'].mean()*100/df1["PATRIMONIO"].mean()
I=df1['INVESTIMENTO'].mean()*100/df1["PATRIMONIO"].mean()

#printing the resulting percentages
print(f'liquidità:{round(L,4)}%')
print(f'risparmio:{round(R,4)}%')
print(f'Investimento:{round(I,4)}%')

# Creating donut chart showing the distribution of the three product categories based on the calculated percentages
labels0 = 'LIQUIDITÁ', 'RISPARMIO', 'INVESTIMENTO'
sizes0 = [L, R, I]
colors=['#cc99ff','#5900b3','#8600b3']
fig0, ax0 = plt.subplots()
ax0.pie(sizes0, labels=labels0, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 15})
ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# draw the inner circle
centre_circle = plt.Circle((0, 0), 0.80, fc='white')
fig0 = plt.gcf()
plt.text(0, 0, "67Mila€", ha='center', va='center', fontsize=42)

# Adding the inner circle in Pie chart
fig0.gca().add_artist(centre_circle)
plt.title('RIPARTIZIONE DEL PATRIMONIO SUI 3 MACROAGGREGATI DI PRODOTTO', fontsize=15)
plt.show()

# Define a list containing all products 
ind=["FL_CCBP", "FL_LIB", "FL_BFP", "FL_TITOLI", "FL_FONDI",
            "FL_PREVIDENZA", "FL_PERSONA", "FL_RAMO_I", "FL_RAMO_III",
            "FL_RAMO_IV", "FL_MULTIRAMO", "FL_POSTEPAY", "FL_EVOLUTION"]

#create a list that contains the counts of each product
fre=[]
for i in ind:  
    fre.append((df1[i] == 1).sum())

#importing pyplot from matplotlib as plt    
import matplotlib.pyplot as plt

# Create a bar chart showing the counts of each product category
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("POSSESSO PRODOTTI")
plt.ylabel('COUNT')
plt.xlabel('PRODOTTI')
plt.bar(ind, fre)
plt.show()

# Define lists of product categories for each of the three macro categories
pdt_liq = ["PATRIMONIO_CONTI", "PATRIMONIO_LIBRETTI", "PATRIMONIO_POSTEPAY", "PATRIMONIO_EVOLUTION"]
pdt_risp = ['PATRIMONIO_BFP']
pdt_inv = ['PATRIMONIO_TITOLI','PATRIMONIO_FONDI','PATRIMONIO_RAMO_I','PATRIMONIO_RAMO_III','PATRIMONIO_RAMO_IV','PATRIMONIO_MULTIRAMO','PATRIMONIO_PREVIDENZA','PATRIMONIO_PERSONA']

#Creating a piechart showing the distribution of Investors (Investitori) and not Investorn (Non Investitori) in the sample
ind=["INVESTITORI",'NON INVESTITORI']
fre=[49.734,50.266]  
labels1 = ind
sizes1 = fre
fig1, ax1 = plt.subplots()
ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'fontsize': 20})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# printing the number of Investors and not investors in the Dataset
df1["FL_INVESTIMENTO"].value_counts()

# Create a new DataFrame called "dfinv" containing only the rows of "df1" where the value in the "FL_INVESTIMENTO" column is equal to 1
#So containing only Investors
dfinv=df1[df1["FL_INVESTIMENTO"]==1]

# Print the number of rows in the "dfinv" DataFrame
dfinv.shape[0]

# Calculating the percentages allocated by investors to each investment product
print('percentuali allocate sui singoli prodotti di investimento dagli investitori:\n')
percinv=[]
for i in pdt_inv:
    percinv.append(round(dfinv[i].mean()*100/dfinv['INVESTIMENTO'].mean(),2))
percpdt=list(zip(pdt_inv,percinv))
print(percpdt)

#Creating a barplot representing the percentage allocated by investors to each investment product
labels = pdt_inv
sizes = percinv
plt.rcParams["figure.figsize"] = [25.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("PATRIMONIO VS PRODOTTI", fontsize=15)
plt.ylabel('% DI PATRIMONIO ALLOCATO', fontsize=15)
plt.xlabel('PRODOTTI D\'INVESTIMENTO',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.bar(labels, sizes)
plt.show()

# COMPARISONS BETWEEN MALE AND FEMALE CUSTOMERS


#Creating a dataframe from the original one which contains only female customers
dff = df1[df1['SESSO'] == "F"]  

#Creating a dataframe from the original one which contains only male customers
dfm = df1[df1['SESSO'] == "M"]  


# In[118]:


#Creating a list contaning all the numerical variables
numericv = list(df1.select_dtypes(include=['float64']).columns)
numericv.append('CLASSI_ETA')
numericv.append('PROPENSIONE_AL_RISCHIO')

#Creating a list containing the wealth allocated by customers on each product 
patrimoniopdt=[i for i in numericv if i.startswith("PATRIMONIO_")]


# In[119]:


#Creating a list containing all the categorical variables
columns = list(df1.columns)
categoricv = [i for i in columns if i not in numericv]


# In[120]:


#printing main descriptive statistics for the numerical variables in the male customers' dataset
dfm[numericv].describe() 


# In[121]:


#Calculating the percentages of wealth allocated by male customers on each product-macrocategory
Lm=dfm['LIQUIDITA'].mean()*100/dfm["PATRIMONIO"].mean()
Rm=dfm['RISPARMIO'].mean()*100/dfm["PATRIMONIO"].mean()
Im=dfm['INVESTIMENTO'].mean()*100/dfm["PATRIMONIO"].mean()
#printing the results
print(f'liquidità:{round(Lm,4)}%')
print(f'risparmio:{round(Rm,4)}%')
print(f'Investimento:{round(Im,4)}%')

#printing the average wealth for the male customers
print(f'Patrimonio totale:{round(dfm["PATRIMONIO"].mean(),4)}')


# In[122]:


# creating a donut chart representing the percentages allocated by male customers on each protuct macrocategory

labels = 'LIQUIDITÁ', 'RISPARMIO', 'INVESTIMENTO'
sizes = [Lm, Rm, Im]
colors=['#b3d1ff',"#0047b3",'#1a75ff']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 30})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# draw the inner circle
centre_circle = plt.Circle((0, 0), 0.80, fc='white')

#Inserting the average wealth in the center
plt.text(0, 0, "69Mila €", ha='center', va='center', fontsize=42)
fig1 = plt.gcf()


# Adding the inner circle in Pie chart
fig1.gca().add_artist(centre_circle)

plt.title('UOMINI', fontsize=30)

plt.show()


# In[123]:


#Creating a Dataframe from the previous one which contains only male investors
dfminv=dfinv[dfinv['SESSO']=="M"]

#Calculating the percentages allocated by male investors on each investment product
print('percentuali allocate sui singoli prodotti di investimento tra gli investitori uomini:\n')
percinvm=[]
for i in pdt_inv:
    percinvm.append(round(dfminv[i].mean()*100/dfminv['INVESTIMENTO'].mean(),3))
print(percinvm)

#creating a barplot representing the percentages allocated by male investors on each investment product   
labels = pdt_inv
sizes = percinvm

plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("PERCENTUALI DI ALLOCAZIONE DEL PATRIMONIO")
plt.ylabel('% Di patrimonio allocato', fontsize=15)
plt.xlabel('Prodotti d\'investimento',fontsize=15)


plt.bar(labels, sizes)

plt.show()


# In[124]:


#creating a barplot showing the number of male customers in the sample for each digital cluster
ax = dfm['CLUSTER_DIGITAL'].value_counts().plot(kind='bar', title ="Digitalizzazione della clientela femminile", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("cluster digital di appartenenza", fontsize=12)
ax.set_ylabel("count", fontsize=12)
#calculating the percentages of male customers in the sample for each digital cluster
print(dfm['CLUSTER_DIGITAL'].value_counts()/dfm.shape[0]*100)


# In[125]:


#printing main descriptive statistics for the numerical variables in the female customers' dataset
dff[numericv].describe()


# In[126]:


#Calculating the percentages of wealth allocated by female customers on each product-macrocategory
Lf=dff['LIQUIDITA'].mean()*100/dff["PATRIMONIO"].mean()
Rf=dff['RISPARMIO'].mean()*100/dff["PATRIMONIO"].mean()
If=dff['INVESTIMENTO'].mean()*100/dff["PATRIMONIO"].mean()

#Printing the resulting percentages
print(f'liquidità:{round(Lf,4)}%')
print(f'risparmio:{round(Rf,4)}%')
print(f'Investimento:{round(If,4)}%')

#printing the average wealth for the female customers
print(f'Patrimonio totale:{round(dff["PATRIMONIO"].mean(),4)}')


# In[127]:


#Creating a donut chart showing the percentages allocated by female customers on each product macrocategory
labels = 'LIQUIDITÁ', 'RISPARMIO', 'INVESTIMENTO'
sizes = [Lf, Rf, If]
colors=['#ffcc99','#cc6600','#ff9933']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 30})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# draw the inner circle
centre_circle = plt.Circle((0, 0), 0.80, fc='white')
fig1 = plt.gcf()

#Adding the average wealth for female customers in the middle of the chart
plt.text(0, 0, "65Mila€", ha='center', va='center', fontsize=42)

# Adding the inner Circle in Pie chart
fig1.gca().add_artist(centre_circle)

plt.title('DONNE', fontsize=30)

plt.show()


# In[128]:


#creating a new dataframe from the previous one containing only female investors
dffinv=dfinv[dfinv['SESSO']=='F']

#calculating the percentages allocated by female investors on each investment product
print('percentuali allocate sui singoli prodotti di investimento tra le investitrici donne:\n')
percinvf=[] 
for i in pdt_inv:
    percinvf.append(round(dffinv[i].mean()*100/dffinv['INVESTIMENTO'].mean(),3))
print(percinvf)

#creating a barplot showing the percentages allocated by female investors on each investment product
labels = pdt_inv
sizes = percinvf

plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("PERCENTUALI DI ALLOCAZIONE DEL PATRIMONIO INVESTITO")
plt.ylabel('% DI PATRIMONIO INVESTITO')
plt.xlabel('PRODOTTI')


plt.bar(labels, sizes)

plt.show()


# In[129]:


#creating a barplot showing the number of female customers for each digital cluster
ax = dff['CLUSTER_DIGITAL'].value_counts().plot(kind='bar', title ="Digitalizzazione della clientela femminile", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("cluster digital di appartenenza", fontsize=12)
ax.set_ylabel("count", fontsize=12)

#printing the percentage of female customers for each digital cluster
print(dff['CLUSTER_DIGITAL'].value_counts()/dff.shape[0]*100)


# In[130]:


#Printing the percentage on multibank and not-multibank female customers in the sample
print(dff.FL_MULTIBANCA.value_counts()/dff.shape[0]*100)

#Printing the percentage on multibank and not-multibank male customers in the sample
print(dfm.FL_MULTIBANCA.value_counts()/dfm.shape[0]*100)


# In[131]:


#printing the rounded average wealth for female customers
print(round(dff['PATRIMONIO'].mean(),2))

#printing the rounded average wealth for male customers
print(round(dfm['PATRIMONIO'].mean(),2))


# In[132]:


#Comparing percentages allocated by male and female investors on investment products 
print('Confronto percentuali allocazione prodotti:\n')
for i in pdt_inv:
    diff =(dfminv[i].mean()*100/dfminv['INVESTIMENTO'].mean())-(dffinv[i].mean()*100/dffinv['INVESTIMENTO'].mean())
    print(f'{i}: {round(diff,4)}%')


# Da una prima analisi emerge come in percentuale uomini e donne allochino il proprio patrimonio in modo similare sui differenti prodotti di investimento.
# 
# Fondi, RamoIII e Multiramo sono i prodotti a cui corrisponde un maggior rischio e un maggior rendimento. Notiamo che la percentuale del patrimonio dedicata a tali prodotti è in media leggermente più elevata per gli uomini.
# In percentuale la quota allocata dalle donne è in media superiore per la Ramo I e la ramo IV, prodotti a cui è associato un basso livello di rischio e rendimento.

# In[133]:


#Comparing the averages values of numerical features in male and female customers' dataframes 
round(dfm[numericv].mean()-dff[numericv].mean(),4)


# Confrontando gli importi medi allocati notiamo alcune differenze imputabili però al patrimonio (superiore negli uomini rispetto alle donne)

# grafici di confronto delle percentuali uomini vs donne

# In[134]:


#Creating a multiple barplot showing the percentages allocated by male and female investors on each investment product
figure2 = plt.bar(pdt_inv, percinvm, align='edge', width=-0.4, label='Uomini')
figure1 = plt.bar(pdt_inv, percinvf, align='edge', width=0.4, label='Donne')
plt.legend(fontsize=25)
plt.title("ALLOCAZIONE DEL PATRIMONIO DEDICATO AGLI INVESTIMENTI",fontsize=15)
plt.xlabel('PRODOTTI DI INVESTIMENTO',fontsize=15)
plt.ylabel('% DI PATRIMONIO ALLOCATA',fontsize=15)
plt.rcParams["figure.figsize"] = [25.00, 11.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[135]:


#Printing the percentages of female investors by risk-propensity
print(f"investitori:\n{dfminv['PROPENSIONE_AL_RISCHIO'].value_counts()/dfminv.shape[0]*100}")

#Printing the percentages of male investors by risk-propensity
print(f"investitrici:\n{dffinv['PROPENSIONE_AL_RISCHIO'].value_counts()/dffinv.shape[0]*100}")

#Creating a multiple barplot showing the percentage of male and female investors by risk-propensity  
uomini = [2.838454,41.934637,45.883167,8.660985,0.682757]
donne = [3.112246,44.847948,44.191459,7.369006,0.479341]
propensione = [1,2,3,4,5]
figure2 = plt.bar(propensione, uomini, align='edge', width=-0.4, label='Uomini')
figure1 = plt.bar(propensione, donne, align='edge', width=0.4, label='Donne')
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [15.00, 11.50]
plt.title("PROPENSIONE AL RISCHIO UOMINI VS DONNE",fontsize=15)
plt.xlabel('PROFILO DI RISCHIO',fontsize=15)
plt.ylabel("% DI INVESTITORI",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[136]:


#printing the number of male investors by risk propensity
dfminv['PROPENSIONE_AL_RISCHIO'].value_counts()/dfminv.shape[0]*100


# In[137]:


#printing the number of female investors by risk propensity
dffinv['PROPENSIONE_AL_RISCHIO'].value_counts()/dffinv.shape[0]*100


# In[138]:


#Creating a multiple barplot showing the percentage of male and female customers in each digital cluster
uomini = [41.599758,8.159603,18.595085,28.306900,3.338655]
donne = [46.248697,8.711149,19.893683,23.188279,1.958191]
cluster = ["D0","D1","D2","D3","D4"]
figure2 = plt.bar(cluster, uomini, align='edge', width=-0.4, label='Uomini')
figure1 = plt.bar(cluster, donne, align='edge', width=0.4, label='Donne')
plt.legend(fontsize=30)
plt.title("CLUSTER DIGITAL VS SESSO",fontsize=15)
plt.xlabel('CLUSTER DIGITAL',fontsize=15 )
plt.ylabel('% SAMPLE',fontsize=15)
plt.show()


# In[139]:


#creating a multiple barplot showing the percentage of male and female customers for each age class
uomini = [2.735017, 8.13047, 11.878154, 17.031219, 20.292963, 20.212555, 19.719623]
donne = [1.802307, 5.882456, 10.643068, 18.067731, 22.14175, 20.50934, 20.953348]

cluster = ['18-24','25-34','35-44','45-54','55-64','65-74','75+']

figure2 = plt.bar(cluster, uomini, align='edge', width=-0.4, label='Uomini')
figure1 = plt.bar(cluster, donne, align='edge', width=0.4, label='Donne')
plt.legend(fontsize=30)
plt.title("CLASSI D'\ETÁ VS SESSO",fontsize=15)
plt.ylabel('% SAMPLE',fontsize=15)
plt.xlabel('CLASSI D\'ETÁ',fontsize=15)

plt.show()



# In[140]:


#calculating and printing the average age for female customers
etaf_count=list(dff.CLASSI_ETA.value_counts())

valf_centr=[(55+64)/2,(65+74)/2,(75+90)/2,(45+54)/2,(35+44)/2,(25+34)/2,(18+24)/2]

ponderaf = list(zip(valf_centr,etaf_count))

ponderataf=[]
for i in ponderaf:
    ponderataf.append((i[0]*i[1])/dff.shape[0])
print(f'età media donne: {round(sum(ponderataf),2)}')

#calculating and printing the average age for male customers
etam_count=list(dfm.CLASSI_ETA.value_counts())

valm_centr=[(55+64)/2,(65+74)/2,(75+90)/2,(45+54)/2,(35+44)/2,(25+34)/2,(18+24)/2]

ponderam = list(zip(valm_centr,etam_count))

ponderatam=[]
for i in ponderam:
    ponderatam.append((i[0]*i[1])/dfm.shape[0])
print(f'età media uomini: {round(sum(ponderatam),2)}')


# In[142]:


#calculating the percentages of male customers for each geographic area
(dfm['AREA_TERRITORIALE'].value_counts()/dfm.shape[0]*100)


# In[143]:


#calculating the percentages of female customers for each geographic area
dff['AREA_TERRITORIALE'].value_counts()/dff.shape[0]*100


# In[144]:


#creating a multiple barplot showing the percentages of male and female customers for each geographic area
uomini = (17.016070,20.832508,26.733712,16.518476,10.307296,8.591938)
donne = (16.940632,21.397357,25.766506,16.793505,10.666713,8.435286)
area = ['MA CENTRO NORD', 'MA NORD OVEST', 'MA SUD', 'MA CENTRO',
       'MA NORD EST', 'MA SICILIA']
figure2 = plt.bar(area, uomini, align='edge', width=-0.4, label='Uomini')
figure1 = plt.bar(area, donne, align='edge', width=0.4, label='Donne')
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [20.00, 11.50]
plt.title("SESSO VS PROVENIENZA GEOGRAFICA",fontsize=15)
plt.xlabel('MACRO AREE TERRITORIALI',fontsize=15)
plt.ylabel('% SAMPLE',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# In[145]:


#printing the total wealth for female customers
print(dff['PATRIMONIO'].sum())

#printing the total wealth for male customers
print(dfm['PATRIMONIO'].sum())


# In[205]:


#creating a barplot showing the total amount invested by female customers on each investment product
totpatrimonioinv=[] 
for i in pdt_inv:
    totpatrimonioinv.append(dff[i].sum())

 
labels = pdt_inv
sizes = totpatrimonioinv

plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.rcParams["figure.autolayout"] = True
plt.title("ALLOCAZIONE DEL PATRIMONIO INVESTITO")
plt.ylabel('PATRIMONIO IN €')
plt.xlabel('PRODOTTI')


plt.bar(labels, sizes)

plt.show()


# In[201]:


#calculating the total amount allocated by female customers of each product of the macrocategory "liquidity"
totpatrimonioliq=[] 
for i in pdt_liq:
    totpatrimonioliq.append(dff[i].sum())

#calculating the total amount allocated by female customers of each of the macrocategory "Savings"
totpatrimoniorisp=[]
for i in pdt_risp:
    totpatrimoniorisp.append(dff[i].sum())


# In[153]:


#Creating a multiple bar plot showing the distribution of the risk-propensity variable across different age groups ("CLASSI_ETA") for female investors
ax = dffinv.groupby(['CLASSI_ETA']).PROPENSIONE_AL_RISCHIO.value_counts().reset_index(name='counts').pivot(index='CLASSI_ETA', columns='PROPENSIONE_AL_RISCHIO', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("CLASSI D'ETÀ", fontsize=15)
ax.set_ylabel("COUNTS", fontsize=15)
plt.title('PROPENSIONE AL RISCHIO VS CLASSI D\'ETÁ', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[154]:


#Creating a bar plot showing the distribution of the average wealth variable across different age groups ("CLASSI_ETA") for female investors
ax1 = dffinv.groupby('CLASSI_ETA')['PATRIMONIO'].mean().plot(kind='bar', title ="PATRIMONIO MEDIO VS CLASSI D'ETÁ", figsize=(15, 10), fontsize=12)
ax1.set_xlabel("CLASSI D'ETÁ", fontsize=12)
ax1.set_ylabel("PATRIMONIO IN €", fontsize=12)


# In[155]:


#Creating a multiple bar plot showing the distribution of the age groups variable across different digital clusters for female investors

ax = dffinv.groupby(['CLASSI_ETA']).CLUSTER_DIGITAL.value_counts().reset_index(name='counts').pivot(index='CLASSI_ETA', columns='CLUSTER_DIGITAL', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("CLASSI D'ETÀ", fontsize=15)
ax.set_ylabel("COUNTS", fontsize=15)
plt.title('CLUSTER DIGITAL VS CLASSI D\'ETÁ', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[72]:


#Creating a multiple bar plot showing the distribution female investors across different age groups.
ax = dff.groupby(['CLASSI_ETA']).FL_INVESTIMENTO.value_counts().reset_index(name='counts').pivot(index='CLASSI_ETA', columns='FL_INVESTIMENTO', values='counts').plot(kind='bar',width = 0.9)
ax.set_xlabel("CLASSI D'ETÀ", fontsize=15)
ax.set_ylabel("Counts", fontsize=15)
plt.title('Chi investe per classi d\'età', fontsize=15)
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = [17.00, 11.50]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[73]:


#creating a list containing product associated to high financial risk
pdt_rischiosi=["PATRIMONIO_FONDI",'PATRIMONIO_RAMO_III','PATRIMONIO_MULTIRAMO']
#creating a list containing product associated to low financial risk
pdt_nonrischiosi=['PATRIMONIO_TITOLI','PATRIMONIO_RAMO_I','PATRIMONIO_RAMO_IV','PATRIMONIO_PREVIDENZA','PATRIMONIO_PERSONA']


# In[74]:


#creating a new variable representing the total ammount allocated by each male investor on high financial risk's product
dfminv['PATRIMONIO_RISCHIO']=dfminv['PATRIMONIO_FONDI']+dfminv["PATRIMONIO_RAMO_III"]+dfminv["PATRIMONIO_MULTIRAMO"]

#creating a new variable representing the total ammount allocated by each male investor on low financial risk's product
dfminv['PATRIMONIO_NON_RISCHIO']=dfminv["PATRIMONIO_TITOLI"]+dfminv["PATRIMONIO_RAMO_I"]+dfminv['PATRIMONIO_RAMO_IV']+dfminv["PATRIMONIO_PREVIDENZA"]+dfminv['PATRIMONIO_PERSONA']


# In[85]:


#calculating the total ammount allocated by male investors on high and low risk products
m_tot_r = dfminv['PATRIMONIO_RISCHIO'].sum()
m_tot_nr = dfminv['PATRIMONIO_NON_RISCHIO'].sum()
print(m_tot_r)
print(m_tot_nr)

#calculating the average ammount allocated by male investors on high and low risk products
m_mean_r = m_tot_r/dfminv.shape[0]
m_mean_nr = m_tot_nr/dfminv.shape[0]
print(m_mean_r)
print(m_mean_nr)

#calculating the percentages allocated by male investors on high and low risk products
m_tot_perc_r = m_tot_r/dfminv['INVESTIMENTO'].sum()*100
m_tot_perc_nr = m_tot_nr/dfminv['INVESTIMENTO'].sum()*100
print(m_tot_perc_r)
print(m_tot_perc_nr)


# In[76]:


#creating a new variable representing the total ammount allocated by each female investor on high financial risk's product
dffinv['PATRIMONIO_RISCHIO']=dffinv['PATRIMONIO_FONDI']+dffinv["PATRIMONIO_RAMO_III"]+dffinv["PATRIMONIO_MULTIRAMO"]
#creating a new variable representing the total ammount allocated by each female investor on low financial risk's product
dffinv['PATRIMONIO_NON_RISCHIO']=dffinv["PATRIMONIO_TITOLI"]+dffinv["PATRIMONIO_RAMO_I"]+dffinv['PATRIMONIO_RAMO_IV']+dffinv["PATRIMONIO_PREVIDENZA"]+dffinv['PATRIMONIO_PERSONA']


# In[77]:


#calculating the total ammount allocated by female investors on high and low risk products
f_tot_r = dffinv['PATRIMONIO_RISCHIO'].sum()
f_tot_nr = dffinv['PATRIMONIO_NON_RISCHIO'].sum()
print(f_tot_r)
print(f_tot_nr)

#calculating the average ammount allocated by female investors on high and low risk products
f_mean_r = f_tot_r/dffinv.shape[0]
f_mean_nr = f_tot_nr/dffinv.shape[0]
print(f_mean_r)
print(f_mean_nr)

#calculating the percentages allocated by female investors on high and low risk products
f_tot_perc_r = f_tot_r/dffinv['INVESTIMENTO'].sum()*100
f_tot_perc_nr = f_tot_nr/dffinv['INVESTIMENTO'].sum()*100
print(f_tot_perc_r)
print(f_tot_perc_nr)


# In[78]:


#calculating the number of female investors for each age class
dffinv['CLASSI_ETA'].value_counts()


# In[79]:


#calculating the number of female investors for each geographic area
dffinv['AREA_TERRITORIALE'].value_counts()


# In[80]:


#calculating the average investment for female investors
dffinv['INVESTIMENTO'].mean()

