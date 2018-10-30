
# coding: utf-8

# In[280]:


import pandas as pd
import matplotlib as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
get_ipython().magic('matplotlib inline')


# In[281]:


train = pd.read_csv("data/train.csv",sep=',')
test = pd.read_csv("data/test.csv",sep=',')


#  ## Visualizacao basica dos dados

# In[286]:


train.head(5)


# ### Verificacao de outliers

# In[283]:


train.boxplot(column = ['preco'])
print("{} linhas no dataser".format(len(train.preco)))


# In[302]:


train_without_outliers = train[np.abs(train.preco - train.preco.mean()) <= (2 * train.preco.std())]
train_without_outliers.preco.hist()
print("{} linhas no dataset".format(len(train_without_outliers.preco)))


# In[303]:


train_without_outliers.boxplot(column = ['preco'])


# ### One hot encoding

# In[304]:


non_discrete_categorical_columns = ['area_util','area_extra']
discrete_categorical_columns = ['quartos','suites','vagas']
binary_columns = ['estacionamento','piscina','playground','quadra','s_festas','s_jogos','s_ginastica','s_ginastica','sauna','vista_mar']


one_hot_data = train_without_outliers.copy()

hot_columns = pd.get_dummies(one_hot_data['tipo'], prefix= 'tipo')
one_hot_data = pd.concat([one_hot_data, hot_columns], axis=1)
one_hot_data = one_hot_data.drop(columns=['tipo'])

for column in discrete_categorical_columns:   
    hot_columns = pd.get_dummies(one_hot_data[column], prefix= column)
    one_hot_data = pd.concat([one_hot_data, hot_columns], axis=1)
    one_hot_data = one_hot_data.drop(columns=[column])

df = one_hot_data.drop(columns=['Id','diferenciais','tipo_vendedor'])
df.head(3)


# In[270]:


## Antes de retirar oluna totoal, dividir as demais por ela, de modoa ter uma proporcao
renda_bairros = pd.read_csv("data/renda_bairros.csv",sep=',')
renda_bairros = renda_bairros.drop(columns=['Total ¹'])
renda_bairros.head(3)


# In[275]:


woth_column_names = ['meio_salario','meio_1_salario','1_2_salarios','2_5_salarios','5_10_salarios','10_20_salarios','20_salarios','sem_salario']
df_renda = pd.concat([df,df.reindex(columns = woth_column_names)], axis = 1)
df_renda = df_renda[df_renda.bairro != 'Beira Rio']
df_renda = df_renda[df_renda.bairro != 'Centro']
for bairro in list(df_renda.bairro.unique()):
    df_renda.loc[df.bairro == bairro, woth_column_names] = renda_bairros[renda_bairros['bairro'] == bairro].iloc[:,1:].values

df_renda = df_renda.drop(columns=['bairro'])
df_renda.head(5)

