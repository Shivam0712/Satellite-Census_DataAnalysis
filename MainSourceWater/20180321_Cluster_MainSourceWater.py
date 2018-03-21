
# coding: utf-8

# In[1]:


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
import geopandas as gpd
import folium
from branca.colormap import linear
from sklearn.preprocessing import normalize


# In[2]:


###Import primary abstract
df_PA=pd.read_csv("/home/tempuser/Doc/Correlations/2011PA_India.csv", encoding = "ISO-8859-1")


# In[3]:


####Import HH amenities
df_HH=pd.read_csv("/home/tempuser/Doc/Correlations/2011HH_India.csv", encoding = "ISO-8859-1")


# In[4]:


###Creating the ID Variable ="Village Code" +"Ward"
df_PA['NewID']=df_PA['Town/Village'].astype(str)+"-"+df_PA['Ward'].astype(str)
df_HH['NewID']=df_HH['7'].astype(str)+"-"+df_HH['8'].astype(str)

###Combining both the data frames
df_comb=df_PA.merge(df_HH, left_on='NewID', right_on='NewID', how='inner')
df_comb=df_comb.drop_duplicates()

###Deleting redundant dataframes
del df_PA
del df_HH


# In[5]:


###Definning useful columns & Subsetting data frame
col= ['State', 'District', 'Subdistt', 'Town/Village', 'Ward', 'EB', 'Level', 'Name', 'TRU', 'No_HH', 'TOT_P', 'TOT_M', 'TOT_F', 'P_06', 'M_06', 'F_06', 'P_LIT', 'M_LIT', 'F_LIT', 'TOT_WORK_P', 'TOT_WORK_M', 'TOT_WORK_F', 'MAINWORK_P', 'MAINWORK_M', 'MAINWORK_F', 'MAIN_CL_P', 'MAIN_CL_M', 'MAIN_CL_F', 'MAIN_AL_P', 'MAIN_AL_M', 'MAIN_AL_F', 'MAIN_HH_P', 'MAIN_HH_M', 'MAIN_HH_F', 'MAIN_OT_P', 'MAIN_OT_M', 'MAIN_OT_F', 'MARGWORK_P', 'MARGWORK_M', 'MARGWORK_F', 'MARG_CL_P', 'MARG_CL_M', 'MARG_CL_F', 'MARG_AL_P', 'MARG_AL_M', 'MARG_AL_F', 'MARG_HH_P', 'MARG_HH_M', 'MARG_HH_F', 'MARG_OT_P', 'MARG_OT_M', 'MARG_OT_F', 'MARGWORK_3_6_P', 'MARGWORK_3_6_M', 'MARGWORK_3_6_F', 'MARG_CL_3_6_P', 'MARG_CL_3_6_M', 'MARG_CL_3_6_F', 'MARG_AL_3_6_P', 'MARG_AL_3_6_M', 'MARG_AL_3_6_F', 'MARG_HH_3_6_P', 'MARG_HH_3_6_M', 'MARG_HH_3_6_F', 'MARG_OT_3_6_P', 'MARG_OT_3_6_M', 'MARG_OT_3_6_F', 'MARGWORK_0_3_P', 'MARGWORK_0_3_M', 'MARGWORK_0_3_F', 'MARG_CL_0_3_P', 'MARG_CL_0_3_M', 'MARG_CL_0_3_F', 'MARG_AL_0_3_P', 'MARG_AL_0_3_M', 'MARG_AL_0_3_F', 'MARG_HH_0_3_P', 'MARG_HH_0_3_M', 'MARG_HH_0_3_F', 'MARG_OT_0_3_P', 'MARG_OT_0_3_M', 'MARG_OT_0_3_F', 'NON_WORK_P', 'NON_WORK_M', 'NON_WORK_F', '12', '13', '14', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139']
df_comb=df_comb.loc[:,col]


# In[33]:


###Creating new variables
df_comb['Rud']=df_comb['74']+df_comb['75']+df_comb['78']+df_comb['79']+df_comb['80']+df_comb['81']
df_comb['Adv']=df_comb['72']+df_comb['73']
df_comb['Int']=df_comb['76']+df_comb['77']


# ###List of variables to cluster villages
# """
# list_villCol=[]
# for i in range(12,15):
#     list_villCol=list_villCol+[str(i)]
# """

# In[34]:


###List of variables to cluster villages
list_villCol=['Rud','Int','Adv']


# In[35]:


###Print mean and standard deviation of all the selected variables
for i in list_villCol:
    print(i,"; mean: ",df_comb[i].mean(),"; std: ",df_comb[i].std())


# In[ ]:


#list_villCol=['Rud','86','Adv']
#var_list=var_list+ ['Rud','76','Adv','77']
#for i in var_list:
#    print(i, df_comb[i].mean())


# In[36]:


###creating the array for PCA
list_villFull=['State', 'District', 'Subdistt', 'Town/Village', 'Ward', 'EB', 'Level', 'Name', 'TRU']+list_villCol
df_pca=df_comb.loc[:,list_villFull]
X=np.array(df_pca.loc[:,list_villCol])
X=normalize(X,axis=0)


# In[37]:


###descriptive PCA
pca= PCA(svd_solver='full')
pca.fit(X, y=None )
a= pca.explained_variance_ratio_
for i in range(len(a)):
    a[i]=round(a[i],3)
print(a)
print(pd.DataFrame(pca.components_,columns=list_villCol))


# In[38]:


###Creating the array for clustering using transformed variables
X=pca.fit_transform(X)
X=X[:,0:1]
X=normalize(X,axis=0)


# In[39]:


###k means determine k
distortions = []
K = range(1,12)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

###Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[40]:


### Clustering and attaching labels
kmeanModel = KMeans(n_clusters=4).fit(X)
kmeanModel.fit(X)
#df_labels= pd.DataFrame(data={'District':df1.index, 'labels':kmeanModel.labels_})
df_pca['labels']=kmeanModel.labels_


# In[46]:


### boxplot for understanding the clusters
for i in list_villCol:
    plot=ggplot(df_pca,aes(x='labels', y=i))+geom_boxplot()
    plot.show()


# In[42]:


### Changing labels to sort them from underdevloped clusters to developed clusters
df_pca.loc[df_pca['labels']==3,'labels']=4
df_pca.loc[df_pca['labels']==1,'labels']=5
df_pca.loc[df_pca['labels']==2,'labels']=6
df_pca.loc[df_pca['labels']==0,'labels']=7


# In[43]:


### number of villages in each variable
df_pca['labels'].value_counts()


# In[44]:


### Filepath as per the variable to be clustered
fp="/home/tempuser/Doc/Cluster/MainSourceWater/MSW_"


# In[45]:


### Saving the boxplots
j=0
for i in list_villCol:
    plot=ggplot(df_pca,aes(x='labels', y=i))+geom_boxplot()
    plot.save(fp+"Vill_"+str(j)+"_"+i+".png")
    j=j+1


# In[47]:


### Saving the village level details in a csv
df_pca.to_csv(fp+"Village_Level.csv")


# In[48]:


### Creating dataframe for district vectors
df_Dist=pd.crosstab(df_pca["District"],df_pca["labels"],margins=False)
list_distCol=list(df_Dist)
list_distCol2=[]
for i in list_distCol:
    list_distCol2=list_distCol2+[str(i)]
df_Dist.columns=list_distCol2    


# In[49]:


### Creating array for clustering districts
df_Dist['Tot']=0
for i in list_distCol2:
    df_Dist['Tot']=df_Dist['Tot']+df_Dist[i]
for i in list_distCol2:
    df_Dist[i]=df_Dist[i]/df_Dist['Tot']

Y=np.array(df_Dist.loc[:,list_distCol2])
Y=normalize(Y, axis=0)


# In[50]:


###k means determine k
distortions = []
K = range(1,12)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(Y)
    kmeanModel.fit(Y)
    distortions.append(sum(np.min(cdist(Y, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Y.shape[0])

###Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[51]:


### Clustering and attaching labels
kmeanModel = KMeans(n_clusters=4).fit(Y)
kmeanModel.fit(Y)
#df_labels= pd.DataFrame(data={'District':df1.index, 'labels':kmeanModel.labels_})
df_Dist['labels']=kmeanModel.labels_


# In[52]:


### printing descriptive boxplot
df_Dist2=pd.DataFrame()
for i in list_distCol2:
    df_DistTemp=pd.DataFrame()
    
    df_DistTemp['Val']=df_Dist[i]
    df_DistTemp['labels']=df_Dist['labels']
    df_DistTemp['cat']=int(i)
    df_Dist2=df_Dist2.append(df_DistTemp)
ggplot(df_Dist2,aes(x='cat', y='Val'))+geom_boxplot()+facet_wrap('labels')
    


# In[53]:


### Sorting labels from underdeveloped to developed
df_Dist.loc[df_Dist['labels']==2,'labels']=4
df_Dist.loc[df_Dist['labels']==0,'labels']=5
df_Dist.loc[df_Dist['labels']==3,'labels']=6
df_Dist.loc[df_Dist['labels']==1,'labels']=7


# In[54]:


### Count of districts in each clusters
df_Dist['labels'].value_counts()


# In[55]:


### Saving the boxplot
df_Dist2=pd.DataFrame()
for i in list_distCol2:
    df_DistTemp=pd.DataFrame()
    
    df_DistTemp['Val']=df_Dist[i]
    df_DistTemp['labels']=df_Dist['labels']
    df_DistTemp['cat']=int(i)
    df_Dist2=df_Dist2.append(df_DistTemp)
plot= ggplot(df_Dist2,aes(x='cat', y='Val'))+geom_boxplot()+facet_wrap('labels')
plot.save(fp+"Dist.png")


# In[56]:


### Saving the district details in CSV
df_Dist.to_csv(fp+"Dist_Level.csv")


# In[57]:


### Saving the geoJson File
shp_dist = gpd.read_file("/home/tempuser/Doc/Cluster/DistrictShapeFile/new.shx")
df_Dist['Dist']=df_Dist.index
shp_dist = shp_dist.merge(df_Dist, left_on='censuscode', right_on='Dist', how="inner")
shp_dist.to_file(fp+"shp_Dist.json", driver="GeoJSON")


# In[58]:


#Creating base layer of leaflet map
map1 =folium.Map(location=[22, 83], zoom_start=4.5, tiles='Mapbox Bright')


# In[59]:


#Creating color map        
colormap = linear.RdYlGn.scale(
    df_Dist['labels'].min(),
    df_Dist['labels'].max())


# In[60]:


#adding the json layer to map

map1 =map1.add_child(folium.GeoJson(data=open(fp+"shp_Dist.json",encoding = 'utf-8-sig').read(),
                              style_function=lambda x: {'fillColor': colormap(x['properties']['labels']),'color': 'black','weight': 0,'fillOpacity':1.0 }))
map1 =map1.add_child(folium.GeoJson(data=open("/home/tempuser/Doc/Cluster/DistrictShapeFile/2011_State.geojson",encoding = 'utf-8-sig').read(),
                              style_function=lambda x: {'color': 'black','weight': 2,'fillOpacity':1.0 }))
map1.save(outfile=fp+"map.html")


# In[61]:


### Finding the overlap of formed clusters with district categories
df_DistType=pd.read_csv("/home/tempuser/Doc/Cluster/DistrictTypes.csv")
df_Dist=df_Dist.merge(df_DistType, left_on="Dist", right_on="DistCode", how="left")
df_Dist['Type']=df_Dist['Type'].fillna('Others')
pd.crosstab(df_Dist["labels"],df_Dist["Type"],margins=True)

