
#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#regression
import statsmodels.api as sm
#VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Influence test
from statsmodels.stats.outliers_influence import OLSInfluence
#plot residuals
from statsmodels.graphics.regressionplots import plot_leverage_resid2
#Breusch Pagan
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

#Load the female investors' dataframe 
df = pd.read_csv('path', index_col = 0)

#View the columns in DataFrame
df.columns

#Define a set of independent variables
X=df.drop(['INVESTIMENTO'],1)

#Define the dependent variable as "Investimento"
y=df[['INVESTIMENTO']]

#Define a list of non-relevant columns to be dropped
to_drop=['FL_ACCR_STIP_PENS_CC','FL_CCBP','FL_PREVIDENZA','FL_PERSONA',
         'FL_RAMO_III','FL_RAMO_IV','FL_MULTIRAMO','PATRIMONIO_BFP',
         'PATRIMONIO_RAMO_I','FL_RISPARMIO','LIQUIDITA','RISPARMIO',
         'CLUSTER_DIGITAL','FL_MIFID_SCADUTO','FL_MIFID_MAI_PROFILATO','DS_PORTAFOGLIO_PERSONAL',
         'AREA_TERRITORIALE_MA SUD','AREA_TERRITORIALE_MA SICILIA','AREA_TERRITORIALE_MA NORD OVEST',
         'AREA_TERRITORIALE_MA NORD EST','AREA_TERRITORIALE_MA CENTRO NORD','FL_TITOLI', 'FL_FONDI', 'FL_RAMO_I',
       'PATRIMONIO_TITOLI', 'PATRIMONIO_FONDI', 'PATRIMONIO_PREVIDENZA','FL_POSSESSO_EMAIL',
       'FL_APP_BP', 'FL_APP_PPAY', "FL_BFP", "FL_LIB","FL_POSTEPAY","FL_EVOLUTION",
       'PATRIMONIO_PERSONA', 'PATRIMONIO_RAMO_III', 'PATRIMONIO_RAMO_IV', 
       'PATRIMONIO_MULTIRAMO','FL_MIFID_PROFILATO','DS_PORTAFOGLIO_VENDITORI MOBILI',"DS_PORTAFOGLIO_PREMIUM",'REGIONE_Basilicata', 'REGIONE_Calabria', 'REGIONE_Campania',
       'REGIONE_Emilia Romagna', 'REGIONE_Friuli Venezia Giulia', 'SEGMENTO',
       'REGIONE_Lazio', 'REGIONE_Liguria', 'REGIONE_Lombardia',
       'REGIONE_Marche', 'REGIONE_Molise', 'REGIONE_Piemonte',
       'REGIONE_Puglia', 'REGIONE_Sardegna', 'REGIONE_Sicilia',
       'REGIONE_Toscana', 'REGIONE_Trentino Alto Adige', 'REGIONE_Umbria', "PATRIMONIO_RISCHIO","%INV",
       "REGIONE_Val D'Aosta", "REGIONE_Veneto",'%RISCHIO','Cluster']

#drop the columns
X.drop(to_drop,1,inplace=True)

#Create a correlation matrix between the relevant independent variables
plt.figure(figsize = (70,40))
corr = X.corr()
matrix = np.triu(corr)
sns.heatmap(corr, annot=True, mask = matrix)

#Drop variables with high correlation
X.drop('PATRIMONIO_EVOLUTION',1,inplace=True)
X.drop('FL_LIQUIDITA',1,inplace=True)
X.drop('PATRIMONIO_POSTEPAY',1,inplace=True)
X.drop('PATRIMONIO_CONTI',1,inplace=True)
X.drop('PATRIMONIO_LIBRETTI',1,inplace=True)
X.drop('TIPO_INTESTAZIONE_Monointestati',1,inplace=True)

#run a OLS 
model = sm.OLS(y,X)
results_ols_big = model.fit(cov_type='hc0') #'h0' --> heteroscedasticity robust
print(results_ols_big.summary())

#calculate the Variance Inflation Factor 
val = X[list(X.columns[:-2])]
vif_info = pd.DataFrame()
vif_info['VIF'] = [variance_inflation_factor(val.values, i) for i in range(val.shape[1])]
vif_info['Column'] = val.columns
vif_info.sort_values('VIF', ascending=False)

#Influence test
test_class = OLSInfluence(results_ols_big)
test_class.dfbetas[:5, :]

#plot residuals
fig, ax = plt.subplots(figsize=(40, 40))
fig = plot_leverage_resid2(results_ols_big, ax=ax)

#multicollinearity test
np.linalg.cond(results_ols_big.model.exog)

#Breush-Pagan
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(results_ols_big.resid, results_ols_big.model.exog)
lzip(name, test)

