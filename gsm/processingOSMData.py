import os
import bz2
from bs4 import BeautifulSoup as bs
nodeData=[]
wayData=[]
#with bz2.BZ2File("india-latest.osm.bz2","rb") as f:
with open("smallBangaloreArea.osm","rb") as f:
    for curLine in f:
        curLine=curLine.decode("utf-8").strip()
        # A single row Node
        if(curLine[0:5]=='<node' and ('</node' in curLine or '/>' in curLine) ):
            #print("Enter non tagged node")
            curNode=bs(curLine,'lxml')
            curNode=curNode.find('node')
            nodeId=int(curNode['id'])
            nodeLat=float(curNode['lat'])
            nodeLon=float(curNode['lon'])
            nodeData.append([nodeId,nodeLat,nodeLon,[]])
        # A multi row node with tags
        elif(curLine[0:5]=='<node' and ('</node' not in curLine and '/>' not in curLine) ):
            #print("Entered tagged node")
            taggedNodeText=curLine
            curLine=next(f).decode("utf-8").strip()
            while('</node' not in curLine):
                taggedNodeText=taggedNodeText + ' ' + curLine
                curLine=next(f).decode("utf-8").strip()
            taggedNodeText=taggedNodeText + '</node>'
            curNode=bs(taggedNodeText,'lxml')
            curNode=curNode.find('node')
            nodeId=int(curNode['id'])
            nodeLat=float(curNode['lat'])
            nodeLon=float(curNode['lon'])
            tagData=curNode.find_all('tag')
            nodeData.append([nodeId,nodeLat,nodeLon,[[curTag['k'],curTag['v']] for curTag in tagData]])
        # A way
        elif(curLine[0:4]=='<way' and ('</way' not in curLine and '/>' not in curLine) ):
            #print("Entered tagged node")
            taggedWayText=curLine
            curLine=next(f).decode("utf-8").strip()
            while('</way' not in curLine):
                taggedWayText=taggedWayText + ' ' + curLine
                curLine=next(f).decode("utf-8").strip()
            taggedWayText=taggedWayText + '</way>'
            curWay=bs(taggedWayText,'lxml')
            curWay=curWay.find('way')
            wayId=int(curWay['id'])
            wayData.append([nodeId,nodeLat,nodeLon]) 
            tagData=curWay.find_all('tag')
            wayNodeData=curWay.find_all('nd')
            wayData.append([wayId,[int(curNode['ref']) for curNode in wayNodeData],[[curTag['k'],curTag['v']] for curTag in tagData]])
            
print("Cell Execution Completed")
