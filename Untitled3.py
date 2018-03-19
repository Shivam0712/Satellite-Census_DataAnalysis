
# coding: utf-8

# In[24]:


###Import Libraries
import geopandas as gpd
import pandas as pd
from haversine import haversine
import math
import numpy as np
from ggplot import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA


# In[2]:


###Import primary abstract
df_PA=pd.read_csv("/home/tempuser/Doc/Correlations/2011PA_India.csv", encoding = "ISO-8859-1")


# In[3]:


####Import HH amenities
df_HH=pd.read_csv("/home/tempuser/Doc/Correlations/2011HH_India.csv", encoding = "ISO-8859-1")


# In[5]:


#Creating the ID Variable ="Village Code" +"Ward"
df_PA['NewID']=df_PA['Town/Village'].astype(str)+"-"+df_PA['Ward'].astype(str)
df_HH['NewID']=df_HH['7'].astype(str)+"-"+df_HH['8'].astype(str)

df_comb=df_PA.merge(df_HH, left_on='NewID', right_on='NewID', how='inner')
df_comb=df_comb.drop_duplicates()

del df_PA
del df_HH


# In[6]:


col= ['State', 'District', 'Subdistt', 'Town/Village', 'Ward', 'EB', 'Level', 'Name', 'TRU', 'No_HH', 'TOT_P', 'TOT_M', 'TOT_F', 'P_06', 'M_06', 'F_06', 'P_LIT', 'M_LIT', 'F_LIT', 'TOT_WORK_P', 'TOT_WORK_M', 'TOT_WORK_F', 'MAINWORK_P', 'MAINWORK_M', 'MAINWORK_F', 'MAIN_CL_P', 'MAIN_CL_M', 'MAIN_CL_F', 'MAIN_AL_P', 'MAIN_AL_M', 'MAIN_AL_F', 'MAIN_HH_P', 'MAIN_HH_M', 'MAIN_HH_F', 'MAIN_OT_P', 'MAIN_OT_M', 'MAIN_OT_F', 'MARGWORK_P', 'MARGWORK_M', 'MARGWORK_F', 'MARG_CL_P', 'MARG_CL_M', 'MARG_CL_F', 'MARG_AL_P', 'MARG_AL_M', 'MARG_AL_F', 'MARG_HH_P', 'MARG_HH_M', 'MARG_HH_F', 'MARG_OT_P', 'MARG_OT_M', 'MARG_OT_F', 'MARGWORK_3_6_P', 'MARGWORK_3_6_M', 'MARGWORK_3_6_F', 'MARG_CL_3_6_P', 'MARG_CL_3_6_M', 'MARG_CL_3_6_F', 'MARG_AL_3_6_P', 'MARG_AL_3_6_M', 'MARG_AL_3_6_F', 'MARG_HH_3_6_P', 'MARG_HH_3_6_M', 'MARG_HH_3_6_F', 'MARG_OT_3_6_P', 'MARG_OT_3_6_M', 'MARG_OT_3_6_F', 'MARGWORK_0_3_P', 'MARGWORK_0_3_M', 'MARGWORK_0_3_F', 'MARG_CL_0_3_P', 'MARG_CL_0_3_M', 'MARG_CL_0_3_F', 'MARG_AL_0_3_P', 'MARG_AL_0_3_M', 'MARG_AL_0_3_F', 'MARG_HH_0_3_P', 'MARG_HH_0_3_M', 'MARG_HH_0_3_F', 'MARG_OT_0_3_P', 'MARG_OT_0_3_M', 'MARG_OT_0_3_F', 'NON_WORK_P', 'NON_WORK_M', 'NON_WORK_F', '12', '13', '14', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139']


# In[7]:


df_comb=df_comb.loc[:,col]


# In[37]:


X=np.array(df_comb.loc[:,['23', '24', '25', '26', '27', '28', '29', '30', '31']])


# In[8]:


var=['12', '13', '14', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139']

for i in var:
    df_comb[i]=df_comb[i]*df_comb['No_HH']*0.01
    df_comb[i]=df_comb[i].astype(int)


# In[9]:


df_Dist= df_comb.groupby(['District'])['No_HH', 'TOT_P', 'TOT_M', 'TOT_F', 'P_06', 'M_06', 'F_06', 'P_LIT', 'M_LIT', 'F_LIT', 'TOT_WORK_P', 'TOT_WORK_M', 'TOT_WORK_F', 'MAINWORK_P', 'MAINWORK_M', 'MAINWORK_F', 'MAIN_CL_P', 'MAIN_CL_M', 'MAIN_CL_F', 'MAIN_AL_P', 'MAIN_AL_M', 'MAIN_AL_F', 'MAIN_HH_P', 'MAIN_HH_M', 'MAIN_HH_F', 'MAIN_OT_P', 'MAIN_OT_M', 'MAIN_OT_F', 'MARGWORK_P', 'MARGWORK_M', 'MARGWORK_F', 'MARG_CL_P', 'MARG_CL_M', 'MARG_CL_F', 'MARG_AL_P', 'MARG_AL_M', 'MARG_AL_F', 'MARG_HH_P', 'MARG_HH_M', 'MARG_HH_F', 'MARG_OT_P', 'MARG_OT_M', 'MARG_OT_F', 'MARGWORK_3_6_P', 'MARGWORK_3_6_M', 'MARGWORK_3_6_F', 'MARG_CL_3_6_P', 'MARG_CL_3_6_M', 'MARG_CL_3_6_F', 'MARG_AL_3_6_P', 'MARG_AL_3_6_M', 'MARG_AL_3_6_F', 'MARG_HH_3_6_P', 'MARG_HH_3_6_M', 'MARG_HH_3_6_F', 'MARG_OT_3_6_P', 'MARG_OT_3_6_M', 'MARG_OT_3_6_F', 'MARGWORK_0_3_P', 'MARGWORK_0_3_M', 'MARGWORK_0_3_F', 'MARG_CL_0_3_P', 'MARG_CL_0_3_M', 'MARG_CL_0_3_F', 'MARG_AL_0_3_P', 'MARG_AL_0_3_M', 'MARG_AL_0_3_F', 'MARG_HH_0_3_P', 'MARG_HH_0_3_M', 'MARG_HH_0_3_F', 'MARG_OT_0_3_P', 'MARG_OT_0_3_M', 'MARG_OT_0_3_F', 'NON_WORK_P', 'NON_WORK_M', 'NON_WORK_F', '12', '13', '14', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139'].sum()


# In[ ]:


df_Dist[0:10]


# In[10]:


for i in var:
    df_Dist[i]=df_Dist[i]/df_Dist['No_HH']
df_Dist['EMP_AL']=(df_Dist['MAIN_CL_P']+df_Dist['MAIN_AL_P'])/(df_Dist['TOT_P']-df_Dist['P_06'])
df_Dist['EMP_NAL']=(df_Dist['MAIN_HH_P']+df_Dist['MAIN_OT_P'])/(df_Dist['TOT_P']-df_Dist['P_06'])
df_Dist['LIT_RAT']=df_Dist['P_LIT']/df_Dist['TOT_P']
var= var + ['EMP_AL','EMP_NAL','LIT_RAT']


# In[ ]:


df_Dist.describe()


# In[11]:


df1=pd.DataFrame()
df2=pd.DataFrame()

for i in range(len(var)):
    for j in range(i+1,len(var)):
        x = np.array(df_Dist[var[i]])
        y = np.array(df_Dist[var[j]])
        r,p = pearsonr(x,y)
        df2.loc[0,'from']=var[i]
        df2.loc[0,'to']=var[j]
        df2.loc[0,'r']=r
        df2.loc[0,'p']=p
        df1=df1.append(df2)


# In[12]:


df1.to_csv("correlations.csv") 



