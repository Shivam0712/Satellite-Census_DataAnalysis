# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:27:16 2018

@author: Night Fury
"""

###Import Libraries
import geopandas as gpd
import pandas as pd

## DATA PRE-PROCESSING

## ODISHA

###Importing shape file
odisha1 = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/or/or1.geojson")
odisha2 = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/or/or2.geojson")
###Combining Odisha
odisha=odisha1.append(odisha2)
odisha=odisha.reset_index(drop=True)
###Creating ID and OBJID
odisha['ObjID']=odisha.index
odisha['ID']="OD-"+odisha['ObjID'].astype(str)
###Creating Vill to combine with Index
odisha['Vill']=odisha['CEN_2001'].str[-7:]
odisha['Vill']=odisha['Vill'].astype(int)
###Importing OdishaIndex
odishaIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/or/or.csv")
###Merge shapefile and Index
odisha=odisha.merge(odishaIndex, left_on="Vill", right_on='village_code_2001', how="inner") 
odishaFinal=odisha.loc[:,['ID','village_code_2011','village_code_2001','geometry']]

## MAHARASTRA

###Importing shape file
maha1 = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/mh/mh1.geojson")
maha2 = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/mh/mh2.geojson")
###Combining Maha
maha=maha1.append(maha2)
maha=maha.reset_index(drop=True)
maha['CEN_2001']=maha['CEN_2001'].astype(str)
###Creating ID and OBJID
maha['ObjID']=maha.index
maha['ID']="MH-"+maha['ObjID'].astype(str)
###Importing mahaIndex
mahaIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/mh/mh.csv")
mahaIndex['CEN_2001']=mahaIndex['CEN_2001'].astype(str)
###Merge shapefile and Index
mahanew=maha.merge(mahaIndex, left_on="CEN_2001", right_on='CEN_2001', how="inner") 
mahanew=mahanew.drop_duplicates(subset=['ID','village_code_2011','village_code_2001'])
mahaFinal=mahanew.loc[:,['ID','village_code_2011','village_code_2001','geometry']]

## GUJARAT

###Importing shape file
gujarat = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/gj/gj.geojson")
###Creating ID and OBJID
gujarat['ObjID']=gujarat.index
gujarat['ID']="GJ-"+gujarat['ObjID'].astype(str)
###Creating Vill to combine with Index
gujarat['Vill']=gujarat['CENSUS_CODE_2001'].str[-8:]
gujarat=gujarat.loc[gujarat['Vill']!='']
gujarat['Vill']=gujarat['Vill'].astype(int)
###Importing gujaratIndex
gujaratIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/gj/gj.csv")
###Merge shapefile and Index
gujarat=gujarat.merge(gujaratIndex, left_on="Vill", right_on='village_code_2001', how="inner") 
gujaratFinal=gujarat.loc[:,['ID','village_code_2011','village_code_2001','geometry']]

## KERALA

###Importing shape file
kerala = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/kl/kl.geojson")
###Creating ID and OBJID
kerala['ObjID']=kerala.index
kerala['ID']="KL-"+kerala['ObjID'].astype(str)
###Creating Vill to combine with Index
kerala['Vill']=kerala['CEN_2001'].str[-8:]
kerala=kerala.loc[kerala['Vill']!='']
kerala['Vill']=kerala['Vill'].astype(int)
###Importing keralaIndex
keralaIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/kl/kl.csv")
###Merge shapefile and Index
kerala=kerala.merge(keralaIndex, left_on="Vill", right_on='village_code_2001', how="inner") 
keralaFinal=kerala.loc[:,['ID','village_code_2011','village_code_2001','geometry']]

## BIHAR

###Importing shape file
bihar = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/br/br.geojson")
###Creating ID and OBJID
bihar['ObjID']=bihar.index
bihar['ID']="BR-"+bihar['ObjID'].astype(str)
###Creating Vill to combine with Index
bihar['Vill']=bihar['CEN_2001'].str[-8:]
bihar=bihar.loc[bihar['Vill']!='']
bihar=bihar.loc[bihar['Vill']!='Pandaul']
bihar['Vill']=bihar['Vill'].astype(int)
###Importing biharIndex
biharIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/br/br.csv")
###Merge shapefile and Index
bihar=bihar.merge(biharIndex, left_on="Vill", right_on='village_code_2001', how="inner") 
biharFinal=bihar.loc[:,['ID','village_code_2011','village_code_2001','geometry']]

## Karnataka

###Importing shape file
kar = gpd.read_file("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/ka/ka.geojson")
###Creating ID and OBJID
kar['ObjID']=kar.index
kar['ID']="KR-"+kar['ObjID'].astype(str)
###Creating Vill to combine with Index
kar=kar.loc[kar['V_CT_CODE']!='',]
kar['V_CT_CODE']=kar['V_CT_CODE'].astype(int)
###Importing karIndex
karIndex=pd.read_csv("X:/Projects/IITDelhi/VillageShapeFile/indian_village_boundaries-master/ka/ka.csv")
kar=kar.merge(karIndex, left_on='V_CT_CODE', right_on='village_code_2001', how="inner") 
karFinal=kar.loc[:,['ID','village_code_2011','village_code_2001','geometry']]

##FINAL STEPS

###Collating all files
dataFinal=biharFinal
dataFinal=dataFinal.append(gujaratFinal)
dataFinal=dataFinal.append(karFinal)
dataFinal=dataFinal.append(keralaFinal)
dataFinal=dataFinal.append(mahaFinal)
dataFinal=dataFinal.append(odishaFinal)

#Save shapefile to geojson
dataFinal.to_file("X:\Projects\IITDelhi\VillageShapeFile\SelectedStatesVillages.json", driver="GeoJSON")
