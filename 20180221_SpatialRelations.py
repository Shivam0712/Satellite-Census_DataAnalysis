###Import Libraries
import geopandas as gpd
import pandas as pd
from haversine import haversine
import math
import numpy as np

## DATA PRE-PROCESSING

###Importing shape file
geodf = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/br/centroids.geojson")

###Removing unneccesary rows
lis=[22916,23138,23663,23727,23814,35493,41108,45641,45643]
lis= [item for item in geodf['ObjID'] if item not in lis]
geodf=geodf.loc[lis]
geodf=geodf.reset_index(drop=True)
del lis

###Importing the main data file
df_PABihar=pd.read_csv("X:/Projects/IITDelhi/02_Final Census Data/2011/2011PA_Bihar1.csv", encoding = "ISO-8859-1")

###Importing the index file
df_BiharIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/br/Book1.csv", encoding = "ISO-8859-1")

###Importing district Lat Lon
df_DistLL=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/br/20180221_BiharDistrictLatLon.csv", encoding = "ISO-8859-1")

###merge PABihar with village centroids
geodf1=geodf.loc[:,['ObjID','geometry']]
df_BiharIndex1=df_BiharIndex.loc[:,['ObjID','village_code_2011']]
df_PABihar=df_PABihar.merge(df_BiharIndex1, left_on="Town/Village", right_on="village_code_2011", how="left")
df_PABihar=df_PABihar.merge(geodf1, left_on="ObjID", right_on="ObjID", how="inner")
df_PABihar=df_PABihar.merge(df_DistLL, left_on="District", right_on='District', how="left")

###extracting coordinates of village centroid from geometry
df_PABihar['VilLon']=df_PABihar['geometry'].apply(lambda p: p.x)
df_PABihar['VilLat']=df_PABihar['geometry'].apply(lambda p: p.y)

###Calculating eucledian distance of every point from district center
def dist(row):
    dist= haversine((row['Dis_Lat'],row['Dis_Lon']),(row['VilLat'],row['VilLon']))
    return dist
df_PABihar['VilDist']=df_PABihar.apply(lambda row: dist(row), axis=1)

## ANALYSIS: START

#Eligible Workforce: Total - Below 6
df_PABihar['EligibleWorkforce']=df_PABihar['TOT_P']-df_PABihar['P_06']

#Formal Workforce: Main_HH + Main_OT
df_PABihar['FormalWorkforce']=df_PABihar['MAIN_HH_P']+df_PABihar['MAIN_OT_P']

def MA(data, Criteria, CriMin, CriMax, CriBucket, val, func):
    min_new=math.floor(CriMin)
    max_new=math.ceil(CriMax)
    df_Temp=pd.DataFrame()
    for i in range(min_new,max_new+1-CriBucket):
        df_Temp.loc[i,'Bucket']=i
        df_Temp.loc[i,'Min']=i
        df_Temp.loc[i,'Max']=i+CriBucket
    return df_Temp

val=0
func=0        
new_df=MA(df_PABihar,'Dist',0,100,3,val,func)
del new_df
