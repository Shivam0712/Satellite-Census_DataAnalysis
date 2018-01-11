# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:38:59 2017

@author: Night Fury
"""

#import the txt file
import csv
import pandas as pd
import math
import numpy as np

#open the paths raw file
with open('X:/Projects/IITDelhi/Paths/detailed_paths_man_correct.txt') as inputfile:
    results1 = pd.DataFrame(data = list(csv.reader(inputfile)))
#open the paths raw file
with open('X:/Projects/IITDelhi/Paths/detailed_paths_politician_final_correct.txt') as inputfile:
    results2 = pd.DataFrame(data = list(csv.reader(inputfile)))
#open the paths raw file
with open('X:/Projects/IITDelhi/Paths/detailed_paths_comp_correct2.txt') as inputfile:
    results3 = pd.DataFrame(data = list(csv.reader(inputfile)))
    
#Combining all the input files
results= results1.copy(deep=True)   
results= results.append(results2)
results= results.append(results3)

#remove duplicates
results=pd.DataFrame(data={'Path':results[0].unique()})

#List of important nodes
impNodes=['00113886','01010145','00017934','00016839','00029033','207278','169183','195022','169443','194703','U70100MH1979PTC020933','L70200MH2007PLC166818','U45202MH2007PTC175973','U70101MH2006PTC192622','U45202MH2010PTC208793','U15990MH1987PTC044777','U45200MH1978PTC020735']

#function for checking if a node is present in path
def presence(row,imp):
    if imp in row[0]:
        return 1
    else:
        return 0

#function for subsetting paths based on presence of imp nodes    
subsetResults=pd.DataFrame(index=[], columns=['Path','Present'])
for i in range(len(impNodes)):
    tempResults= results.copy(deep=True)
    tempResults['Present']=tempResults.apply(lambda row: presence(row, impNodes[i]), axis=1)        
    tempResults= tempResults.loc[(tempResults.Present == 1)]
    subsetResults=subsetResults.append(tempResults)
del tempResults                             
subsetResults=pd.DataFrame(data={'Path':subsetResults['Path'].unique()})    

#function for splitting path to list
def pathSplit(row):
    return row[0].split('---')
subsetResults['Path']=subsetResults.apply(lambda row: pathSplit(row), axis=1)

#Defining Function to expand the path
def expandDF(row):
    returnDF=pd.DataFrame(data={'Edge':[]})
    tempList=row[0]
    for i in range(len(tempList)-1):
        tempDf=pd.DataFrame(data={'Edge':[]})
        temp=sorted([tempList[i],tempList[i+1]])
        tempDf.ix[i,0]=temp[0]+":"+temp[1]
        returnDF=returnDF.append(tempDf)
    return returnDF
expandResults=pd.DataFrame(data={'Edge':[]})
for i in range(len(subsetResults)):
    expandResults=expandResults.append(expandDF(subsetResults.iloc[i]))
expandResults=pd.DataFrame(data={'Edge':expandResults['Edge'].unique()})    

#function to split edge into source and destination
def edgeSplit(row):
    t= row[0].split(':')
    return t[0],t[1]   
expandResults['Source'],expandResults['Destination']=zip(*expandResults.apply(lambda row: edgeSplit(row), axis=1))    
    
#Function for depth of node from important node
def nodeDepth(row, impNodes):
    tempMatrix=[]
    for i in range(len(impNodes)):
        tempImp=impNodes[i]
        temp= len(row[0]) * [100]
        try:    
            pos=row[0].index(tempImp)
        except:
            pos=100
        for j in range(len(temp)):
            temp[j]=abs(j-pos)
        tempMatrix.append(temp)
    tempMatrix= np.matrix(tempMatrix)
    val= (tempMatrix.min(0)).tolist()
    return val
subsetResults['NodeDepth']=subsetResults.apply(lambda row: nodeDepth(row, impNodes), axis=1)

#Finding the minimum depth for every node
nodeDepth=pd.DataFrame(data={'Node':[],'Depth':[]})
for i in range(len(subsetResults)):
    row=subsetResults.iloc[i]
    tempDf=pd.DataFrame(data={'Node':row[0],'Depth':row[1]})
    nodeDepth=nodeDepth.append(tempDf)
nodeDepth=nodeDepth.sort("Depth").groupby("Node", as_index=False).first()

#Pruning the network at depth d
eligibleNodes=nodeDepth.loc[(nodeDepth.Depth<=10)]
eligibleNodes=eligibleNodes.reset_index(drop=True)              
eligibleEdges=expandResults.loc[expandResults.Source.isin(eligibleNodes.Node) & expandResults.Destination.isin(eligibleNodes.Node)]
eligibleNodes['index1'] = eligibleNodes.index

#Name & Color
impNodes1=pd.DataFrame(data={'Node':['113886','01010145','00017934','00016839','00029033','207278','169183','195022','169443','194703','U70100MH1979PTC020933','L70200MH2007PLC166818','U45202MH2007PTC175973','U70101MH2006PTC192622'
],'Name':['Kalanithi Maran',' Sunita Goenka',' Asif Balwa',' Shahid Balwa',' Vinod Goenka','A. Raja','Kanimozhi','A. Raja','M. Karunanidhi','Dayanidhi Maran','CONWOOD','DB REALTY','DB HISKY','ALLIANZ INFRATECH'
],'Color':['#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000'
]})

#Mapping name and color
def mapNodeAttr(row, impNodes1):
    try:
        name=impNodes1.Name[impNodes1['Node'] == row['Node']].tolist()[0]
        color=impNodes1.Color[impNodes1['Node'] == row['Node']].tolist()[0]
    except:
        name=""
        color='#CCCCCC'
    return name,color

eligibleNodes['Name'],eligibleNodes['Color']=zip(*eligibleNodes.apply(lambda row: mapNodeAttr(row, impNodes1), axis=1))
                            
#Function for adding the layout to Nodes
def nodesLayout(row, nlen):
    angleR= (2*math.pi)/nlen
    rad= row['index1']*angleR
    x= nlen * math.cos(rad)/ (2 *math.pi)
    y= nlen * math.sin(rad)/ (2 *math.pi)
 #   x= math.cos(rad)
  #  y= math.sin(rad)
    return rad,x,y
eligibleNodes['Rad'],eligibleNodes['x'],eligibleNodes['y']=zip(*eligibleNodes.apply(lambda row: nodesLayout(row, len(eligibleNodes)), axis=1))

#Function for mapping the display position of text
def textPos(row):
    if (row['x']<=0):
        Val='middle left'
    else:
        Val='middle right'
    return Val
eligibleNodes['textPos']=eligibleNodes.apply(lambda row: textPos(row), axis=1)    
    
#Function for adding edges source and destination nodes position
def soDe(row):
    so=eligibleNodes.index[eligibleNodes['Node'] == row['Source']].tolist()[0]
    de=eligibleNodes.index[eligibleNodes['Node'] == row['Destination']].tolist()[0]
    return so,de
eligibleEdges['So'],eligibleEdges['De']=zip(*eligibleEdges.apply(lambda row: soDe(row), axis=1))

#Function for edge importance & color
def depthComb(row):
    a=eligibleNodes.iloc[row['So'],1]
    b=eligibleNodes.iloc[row['De'],1]
    return str(min(a,b))+":"+str(max(a,b))  
eligibleEdges['depthComb']=eligibleEdges.apply(lambda row: depthComb(row), axis=1)
DepthComb=pd.DataFrame(data={'Comb':['0.0:0.0','0.0:1.0','1.0:1.0','1.0:2.0','2.0:2.0','2.0:3.0','3.0:3.0','3.0:4.0']})
DepthComb=DepthComb.sort(['Comb'],ascending=True)
DepthComb['Color']=['#ff0000','#6d8acf','#84a9dd','#d4daff','#d4daff','#d4daff','#d4daff','#d4daff']  

def edgeColor(row)


#PLOTLY BEGINS
import plotly.plotly as py
from plotly.graph_objs import *

#Functions for obtaining beziers curve
class InvalidInputError(Exception):
    pass
def deCasteljau(b,t): 
    N=len(b) 
    if(N<2):
        raise InvalidInputError("The  control polygon must have at least two points")
    a=np.copy(b) #shallow copy of the list of control points 
    for r in range(1,N): 
        a[:N-r,:]=(1-t)*a[:N-r,:]+t*a[1:N-r+1,:]                             
    return a[0,:]
def BezierCv(b, nr=5):
    t=np.linspace(0, 1, nr)
    return np.array([deCasteljau(b, t[k]) for k in range(nr)])

#Parameters & edge colors    
params=[1.2, 1.5, 1.8, 2.1]    
edge_colors=['#d4daff','#84a9dd', '#5588c8', '#6d8acf']

lines=[]# the list of dicts defining   edge  Plotly attributes
edge_info=[]# the list of points on edges where  the information is placed
    

for i in range(len(eligibleEdges)):
    A=np.array([eligibleNodes.iloc[eligibleEdges.iloc[i,3],6] , eligibleNodes.iloc[eligibleEdges.iloc[i,3],7]]  )
    B=np.array([eligibleNodes.iloc[eligibleEdges.iloc[i,4],6] , eligibleNodes.iloc[eligibleEdges.iloc[i,4],7]]  )
    d=abs(eligibleNodes.iloc[eligibleEdges.iloc[i,3],5]-eligibleNodes.iloc[eligibleEdges.iloc[i,4],5])
    if(d > math.pi): 
        K= math.floor(abs((4 * abs(2*math.pi - d)/pi)-0.01))
    else: 
        K= math.floor(abs((4 * d/pi)-0.01))    
    b=[A, A/params[K], B/params[K], B]
    if(eligibleEdges.iloc[i,1] in (impNodes) and eligibleEdges.iloc[i,2] in (impNodes)):
        color='#ff0000'
    else:    
        color=edge_colors[K]
    pts=BezierCv(b, nr=5)
    text=eligibleEdges.iloc[i,1]+' to '+eligibleEdges.iloc[i,2]
    mark=deCasteljau(b,0.9)
    edge_info.append(Scatter(x=mark[0], 
                             y=mark[1], 
                             mode='markers', 
                             marker=Marker( size=0.5,  color=edge_colors),
                             text=text, 
                             hoverinfo='text'
                             )
                    )
    lines.append(Scatter(x=pts[:,0],
                         y=pts[:,1],
                         mode='lines',
                         line=Line(color=color, 
                                  shape='spline',
                                  width=1.0 
                                 ), 
                        hoverinfo='none' 
                       )
                )

trace2=Scatter(x=eligibleNodes['x'],
           y=eligibleNodes['y'],
           mode='markers+text',
           name=eligibleNodes['Node'],
           marker=Marker(symbol='dot',
                         size=15, 
                         color=eligibleNodes['Color'], 
                         line="#000000"
                         ),
           text=eligibleNodes['Name'],
           textposition=eligibleNodes['textPos'],
           #textangle=50   
           )

axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )

width=800
height=850

def make_annotation(anno_text, y_coord):
    return Annotation(showarrow=False, 
                      text=anno_text,  
                      xref='paper',     
                      yref='paper',     
                      x=0,  
                      y=y_coord,  
                      xanchor='left',   
                      yanchor='bottom',
                      textangle= 90,
                      font=Font(size=12)     
                     )

anno_text1='Blue nodes mark the countries that are both contestants and jury members'
anno_text2='Grey nodes mark the countries that are only jury members'
anno_text3='There is an edge from a Jury country to a contestant country '+\
           'if the jury country assigned at least one vote to that contestant'

title="A circular graph associated to Eurovision Song Contest, 2015<br>Data source:"+\
"<a href='http://www.eurovision.tv/page/history/by-year/contest?event=2083#Scoreboard'> [1]</a>"
layout=Layout(title= title,
              font= Font(size=12),
              showlegend=False,
              autosize=False,
              width=width,
              height=height,
              xaxis=XAxis(axis),
              yaxis=YAxis(axis),          
              margin=Margin(l=40,
                            r=40,
                            b=85,
                            t=100,
                          ),
              hovermode='closest'#,
              #annotations=Annotations([make_annotation(anno_text1, -0.07), 
                                  #     make_annotation(anno_text2, -0.09),
                                  #     make_annotation(anno_text3, -0.11)]
                                  #   )
              )
data=Data(lines+edge_info+[trace2])
fig=Figure(data=data, layout=layout)
py.sign_in('shivm', 'V7rWwhWSEo2YE0w3J73A') 
py.plot(fig, filename='Eurovision-15') 
                                
#Write results
results.to_csv('X:/Projects/IITDelhi/Paths/expresults2.csv', index=False)

math.