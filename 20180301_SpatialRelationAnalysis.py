
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
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


###Importing centroids
df_centroids = pd.read_csv("/home/tempuser/Doc/20180301_Spatial_Analysis/SelectedStatesVillagesCentroid.csv")


# In[3]:


###Importing Primary Abstracts
df_PA = pd.read_csv("/home/tempuser/Doc/20180301_Spatial_Analysis/2011PA_SelectedStates.csv", encoding = "ISO-8859-1")


# In[4]:


###Importing District HQ
df_DistLL= pd.read_csv("/home/tempuser/Doc/20180301_Spatial_Analysis/DistrictHQ.csv")


# In[5]:


###Merge Primary Abstract with centroids with districtLL
df_Main = df_PA.merge(df_centroids, left_on="Town/Village", right_on="village_code_2011", how="inner")
df_Main = df_Main.merge(df_DistLL, left_on="District", right_on="DistCode", how= "inner")


# In[6]:


###Calculating eucledian distance of every point from district center
def dist(row):
    dist= haversine((row['Lat'],row['Lon']),(row['CentY'],row['CentX']))
    return dist
df_Main['VilDist']=df_Main.apply(lambda row: dist(row), axis=1)


# In[7]:


###Subsetting useful columns
df_Spatial=df_Main.loc[:,['State_x','District_x','Subdistt','Town/Village','Ward', 'EB', 'Level', 'Name', 'TRU','TOT_P','P_06','MAIN_HH_P','MAIN_OT_P','ID','CentX', 'CentY','VilDist']]


# In[ ]:


###Histogram of Distances
for i in df_Spatial['District_x'].unique():
    df_temp=df_Spatial.loc[df_Spatial['District_x']==i,:]
    plot1=ggplot(df_temp, aes(x='VilDist'))+geom_histogram(binwidth=1, color='Red')
    fp="/home/tempuser/Doc/20180301_Spatial_Analysis/"+str(i)+".png"
    plot1.save(fp)


# In[22]:


###Boxplot of Distances
for i in df_Spatial['State_x'].unique():
    df_temp=df_Spatial.loc[df_Spatial['State_x']==i,:]
    plot1=ggplot(df_temp, aes(x='District_x', y='VilDist'))+geom_boxplot()
    fp="/home/tempuser/Doc/20180301_Spatial_Analysis/Boxplot/"+str(i)+".png"
    plot1.save(fp)


# In[8]:


###Function for Binning

####Simple binning
"""
def binit(start,end, size):
    df_temp=pd.DataFrame(columns=['From','To','Mid'])
    k=start
    i=0
    while k+size < end:
        df_temp.loc[i,'From']=k
        df_temp.loc[i,'To']= k+size
        df_temp.loc[i,'Mid']=k+size/2
        k=k+size/2
        i=i+1
    return df_temp
"""
####Cumulative  binning
def binit(start,end, size):
    df_temp=pd.DataFrame(columns=['From','To','Mid'])
    k=start
    i=0
    while k+size < end:
        df_temp.loc[i,'From']=0
        df_temp.loc[i,'To']= k+size
        df_temp.loc[i,'Mid']=(k+size)/2
        k=k+size
        i=i+1
    return df_temp



# In[10]:


### Make Bins
df_bin=binit(0,1,0.1)


# In[11]:


### Merging bins with districts
df_x= df_bin.copy(deep=True)
df_y= pd.DataFrame(df_Spatial['District_x'].unique())
df_x['key']=1
df_y['key']=1
df_distFinal=df_x.merge(df_y, on='key')
df_distFinal.rename(columns={0:'District'}, inplace=True)
del df_x, df_y, df_distFinal['key']


# In[12]:


### Fill the final data frame 
for i in range(len(df_distFinal)):
    vildist_temp=df_Spatial.loc[df_Spatial['District_x']==df_distFinal.loc[i,'District'],'VilDist']
    floor= vildist_temp.quantile(df_distFinal.loc[i,'From'])
    top= vildist_temp.quantile(df_distFinal.loc[i,'To'])
    df_temp=df_Spatial.loc[(df_Spatial['District_x']==df_distFinal.loc[i,'District']) & (df_Spatial['VilDist']>=floor) & (df_Spatial['VilDist']<=top),:]
    emp= (df_temp['MAIN_HH_P'].sum()+df_temp['MAIN_OT_P'].sum())/(df_temp['TOT_P'].sum()-df_temp['P_06'].sum())
    df_distFinal.loc[i,'Floor']= floor
    df_distFinal.loc[i,'Top']= top
    df_distFinal.loc[i,'Val']= emp
    


# In[13]:


### Extracting the shape out of spatial descent
for i in range(len(df_distFinal)):
    df_temp=df_distFinal.loc[df_distFinal['District']==df_distFinal.loc[i,'District'],'Val']
    df_distFinal.loc[i,'Val1']=(df_distFinal.loc[i,'Val']-df_temp.min())/(df_temp.max()-df_temp.min())


# In[14]:


### Saving all the plots of spatial descent
for i in df_distFinal['District'].unique():
    df_temp=df_distFinal.loc[df_distFinal['District']==i,:]
    #### Plot of spatial descent
    "plot1=ggplot(df_temp, aes('To', 'Val'))+geom_point()+geom_line()+ylim(0,1)"
    #### Plot of spatial descent shape
    plot1=ggplot(df_temp, aes('To', 'Val1'))+geom_point()+geom_line()+ylim(0,1)
    fp="/home/tempuser/Doc/20180301_Spatial_Analysis/spatial_drop/"+str(i)+".png"
    plot1.save(fp)


# In[15]:


###Creating the shape attribute df 
df_ShapeAttr=df_distFinal.pivot(index='District', columns='Mid')['Val1']


# In[16]:


###normalizing variance in each bin before clustering
for i in list(df_ShapeAttr):
    std=df_ShapeAttr[i].std()
    df_ShapeAttr[i]=df_ShapeAttr[i]/std
arr_ShapeAttr=np.array(df_ShapeAttr)


# In[17]:


###k means determine k
distortions = []
K = range(1,12)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(arr_ShapeAttr)
    kmeanModel.fit(arr_ShapeAttr)
    distortions.append(sum(np.min(cdist(arr_ShapeAttr, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / arr_ShapeAttr.shape[0])

###Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[18]:


### Clustering and attaching labels
kmeanModel = KMeans(n_clusters=5).fit(arr_ShapeAttr)
kmeanModel.fit(arr_ShapeAttr)
df_labels= pd.DataFrame(data={'District':df_ShapeAttr.index, 'labels':kmeanModel.labels_})
df_distFinal1=df_distFinal.merge(df_labels, left_on= 'District' ,right_on='District', how='left')


# In[21]:


### Saving all the plots of spatial descent with labels of kmean clustering
for i in df_distFinal1['District'].unique():
    df_temp=df_distFinal1.loc[df_distFinal1['District']==i,:]
    label=df_temp['labels'].unique()[0]
    #### Plot of spatial descent
    plot1=ggplot(df_temp, aes('To', 'Val'))+geom_point()+geom_line()
    fp1="/home/tempuser/Doc/20180301_Spatial_Analysis/clusters_value/"+str(label)+"_"+str(i)+".png"
    plot1.save(fp1)
    #### Plot of spatial descent shape
    plot2=ggplot(df_temp, aes('To', 'Val1'))+geom_point()+geom_line()+ylim(0,1)
    fp2="/home/tempuser/Doc/20180301_Spatial_Analysis/clusters_shape/"+str(label)+"_"+str(i)+".png"
    plot2.save(fp2)


# In[24]:


### Saving all the BoxPlots of spatial descent with labels of kmean clustering
for i in df_distFinal1['labels'].unique():
    df_temp=df_distFinal1.loc[df_distFinal1['labels']==i,:]
    plot3=ggplot(df_temp, aes('To', 'Val1'))+geom_boxplot()
    fp3="/home/tempuser/Doc/20180301_Spatial_Analysis/labels/"+str(i)+".png"
    plot3.save(fp3)


# In[30]:


#Saving data files
df_distFinal1.to_csv("/home/tempuser/Doc/20180301_Spatial_Analysis/labels/data.csv")
df_labels1=df_labels.merge(df_DistLL, left_on="District", right_on="DistCode", how="left")
df_labels1.to_csv("/home/tempuser/Doc/20180301_Spatial_Analysis/labels/labels.csv")


# In[ ]:


#NOT USEFUL###################################################


# In[181]:


df_distFinalSubset=df_distFinal1.loc[df_distFinal1['labels']>1,:]
df_ShapeAttr=df_distFinalSubset.pivot(index='District', columns='Mid')['Val1']


# In[ ]:


#for i in list(df_ShapeAttr):
   std=df_ShapeAttr[i].std()
    df_ShapeAttr[i]=df_ShapeAttr[i]/std
arr_ShapeAttr=np.array(df_ShapeAttr)


# In[257]:


df_distFinal1['labels'].unique()


# In[82]:


X=np.array(df_new)
pca= PCA(svd_solver='full')
pca.fit(X, y=None)


# In[79]:


print(pca.explained_variance_ratio_)


# In[84]:


print(pca.singular_values_)


# In[29]:


df_DistLL

