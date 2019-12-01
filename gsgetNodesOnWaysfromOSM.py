import overpy
import pandas as pd

# I will be writing my own library
def getNodeData(line):
    x = re.search(r'\bid="\d+"', line)
    curNode=x.group().replace("id=",'').replace('"','')
    x = re.search(r'\blat="[^"]+"', line)
    curLat=x.group().replace("lat=",'').replace('"','')
    x = re.search(r'\blon="[^"]+"', line)
    curLon=x.group().replace("lon=",'').replace('"','')
    return([curNode,curLat,curLon])

nodeData=[]
wayData=[]
with open("map.osm","rb") as file:
    line = file.readline()
    while line:
        #print(line)
        line=line.decode("utf-8")
        # Parse the node data
        if("<node" in line):
            nodeData.append(getNodeData(line))
        # Parse the way data
        if("<way" in line):
            x = re.search(r'\bid="\d+"', line)
            curWayId=x.group().replace("id=",'').replace('"','')
            curWayNodes=[]
            line=file.readline()
            line=line.decode("utf-8")
            while("</way" not in line):
                if("<nd ref=" in line):
                    x = re.search(r'\bref="\d+"', line)
                    curNodeId=x.group().replace("ref=",'').replace('"','')
                    curWayNodes.append(curNodeId)
                line=file.readline()
                line=line.decode("utf-8")
            wayData.append([curWayId,curWayNodes])
        line=file.readline()
        
nodeData=pd.DataFrame(nodeData,columns=['nodeId','lat','lon'])
allWayNodes=[y for x in wayData for y in x[1]]
allWayNodes=pd.DataFrame(allWayNodes,columns=['nodeId'])
allWayNodes=[y for x in wayData for y in x[1]]
allWayNodes=pd.DataFrame(allWayNodes,columns=['nodeId'])
allWayNodes=allWayNodes.merge(nodeData,left_on=['nodeId'],right_on=['nodeId'],how='left')
print("Cell Execution Completed")
