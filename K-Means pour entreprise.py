#!/usr/bin/env python
# coding: utf-8

# # Segmentation client 

# 
# ***Objectif est de rassembler les clients en groupes d’individus ayant des caractéristiques similaires ou mêmes profils.
# Stratégie qui permet de fidéliser un groupe spécifique de clients, d’attirer de nouveaux clients***
# 
# Base de donnée disponible sur Kaggle
# 
# 

# ## Segmentation client par K-Means

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install kmodes')


# ### Base de modélisation

# In[8]:


#Lecture des tables
db = pd.read_excel("/Users/jalis/Downloads/customer.xlsx" )


# #### Exploration de la table

# In[12]:


db.head()


# In[47]:


#valeurs manquantes
db.isnull().sum()


# ### Analyse Exploratoire des Données
# 

# **Analyse exploratoire**

# In[13]:


#Description de la table
db.describe()


# **Analyser la distribution de la variable Age**

# In[17]:


#Distribution
plt.figure(figsize=(30, 10))
db.boxplot(column='Annual Income (k$)', vert = False )
plt.title("Distribution de l'âge de client", fontsize = 18)
plt.show()


# 50% gagne en dessous de 60K
# Le revenu minimal est de 15k

# In[18]:


plt.figure(figsize=(10, 4))

print(color.BOLD + "Analyse de l'âge du client" + color.END)

#Distribution
db['Age'].hist(bins = 100);


# 3 grosses fourchettes d'âge. 20-30ans/40-50ans/+50ans

# **Analyser la distribution de la variable Gender**

# In[19]:


sns.countplot(x="Gender", hue=None, data=db)


# **Analyse bivariée**

# In[52]:


plt.figure(figsize=(20, 8))
sb.scatterplot(x=db['Age'], y=db['Annual Income (k$)'], hue=db['Gender']);


# 3 classes différentes, ceux qui ont entre 20 et 30 ans gagnes moins que les autres. Les hommes gagnent plus que les femmes.

# In[53]:


table = pd.pivot_table(db, index='Gender', aggfunc='mean')
table


# La moyenne d'age des femmes est supérieure à celle des hommes. La moyenne des revenus des hommes est supérieur a la celle des femmes, mais les femmes dépensent plus que les hommes.

# **Analyse des correlations**

# In[20]:


sns.heatmap(db.corr(), annot=True, cmap="coolwarm")


# Un lien entre l'age et le revenu annuel

# In[55]:


sb.set()
sb.pairplot(db);


# on peut voir une corrélation entre le revenu et l'age et entre les dépenses et le revenu.

# # 5. Construction du modèle

# ##  Data Processing

# #### base de modélisation

# In[23]:


db_mod = db.copy()


# #### standardisation

# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Le but étant de normaliser les variables


# In[25]:


### 1.2 Analyse Exploratoire des Données
db_mod.iloc[:, 1:] = scaler.fit_transform(db_mod.iloc[:, 1:].to_numpy())

db_mod


# ## Modelisation

# #### Importation de la librairie de K-Means

# In[26]:


get_ipython().system('pip install kmodes')
from kmodes.kprototypes import KPrototypes

# Ignore warnings
#import warnings
#warnings.filterwarnings('ignore', category = FutureWarning)


# #### Identification de la variable catégorielle

# In[27]:


#identification de la variable quali,0 car c'est la colonne 
catColumnsPos = [0]
#catColumnsPos = [db_mod.columns.get_loc(col) for col in list(db_mod.select_dtypes('object').columns)]


# #### Base de modélisation 

# In[29]:


dfMatrix= db_mod.to_numpy()
dfMatrix


# #### Estimation du modèle

# In[33]:


# Modèle avec 5 clusters
kprototype = KPrototypes(n_jobs = -1, 
                         n_clusters = 5, 
                         init = 'Huang', 
                         random_state = 0)

# Estimation du modèle
kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)


# #### Output du modèle

# In[35]:


#Définition d'une étiquette pour stocker les classes ou groupe de clients du modèle
labels = kprototype.labels_
# Affecation
db_mod = db.copy()

db_mod['labels'] = labels
db_mod.head()


# #### Interprétation des résultats

# In[67]:


table = pd.pivot_table(db_mod, index='labels', aggfunc='mean')
table


# 5 groupes différents avec des revenus moyens et des dépenses différentes

# In[37]:


db_mod


# In[39]:


plt.scatter(x=db_mod.loc[:,'Annual Income (k$)'],
            y=db_mod.loc[:,'Spending Score (1-100)'],
            c=db_mod.loc[:,'labels'])
               


# On distingue 5 groupes, le membres du premier groupe, gagne entre 20 et 40k/an et ont un score de 40%, ils dépensent 40% de leurs revenus. On peut voir tout à droite, un groupe caracterisé par des salaires élévés et des dépenses faibles.
# 
# 

# In[49]:


print(kprototype.cluster_centroids_)

print(kprototype.n_iter_)

print(kprototype.cost_)


# ## Pertinence du modèle

# In[66]:


# Modèle
cost = []
kprototype2 = KPrototypes(n_jobs = -1, 
                         n_clusters = 5, 
                         init = 'Huang', 
                         random_state = 0)

# Estimation du modèle
kprototype2.fit_predict(dfMatrix, categorical = catColumnsPos)


# Check the cost of the clusters created
kprototype2.cost_


# L'inertie parait acceptable, on peut garder 4 clusters différents pour classer les clients.

# In[ ]:




