import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

numEstimators=500
X, y = make_classification(n_samples=3000, n_features=1000,n_redundant=0,random_state=0, shuffle=False)
rfModel = RandomForestClassifier(n_estimators=numEstimators, max_depth=8,random_state=0,n_jobs=-1)
rfModel.fit(X, y)  
pickle.dump(rfModel,open("rfModel","wb") )
sizeVal=os.stat('rfModel').st_size
print("Size is {0}".format(sizeVal/(1024*1024))) # 

# We will now see whether we can create a data structure which we do the prediction as well as reduce the memory footprint
# 206 ms
%time results=rfModel.predict_proba(X[0:1])

tree1=[]
tree2=[]
for x in range(numEstimators):
    children_left = rfModel.estimators_[x].tree_.children_left.astype(int)
    children_right = rfModel.estimators_[x].tree_.children_right.astype(int)
    feature = rfModel.estimators_[x].tree_.feature
    threshold = rfModel.estimators_[x].tree_.threshold
    values0=[y[0][0] for y in rfModel.estimators_[x].tree_.value]
    values1=[y[0][1] for y in rfModel.estimators_[x].tree_.value]
    #tree[x]={'left':children_left,'right':children_right,'feature':feature,'threshold':threshold,'values':values}
    if(len(tree)==0):
        tree1=[np.array(list(zip(children_left,children_right,feature)))]
        tree2=[np.array(list(zip(threshold,values0,values1)))]
    else:
        tree1.append(np.array(list(zip(children_left,children_right,feature))))
        tree2.append(np.array(list(zip(threshold,values0,values1))))
        
tree1=np.array(tree1)
tree2=np.array(tree2)

print("Cell Execution Completed")

sizes=[]
for x in range(len(tree1)):
    sizes.append(tree1[x].shape[0])
sizeMax=max(sizes)

tree1_1=np.zeros((500,sizeMax,3),dtype=np.int)
for i,x in enumerate(tree1):
    for j,y in enumerate(x):
        tree1_1[i,j,:]=y
        
tree2_1=np.zeros((500,sizeMax,3),dtype=np.float16)
for i,x in enumerate(tree2):
    for j,y in enumerate(x):
        tree2_1[i,j,:]=[np.round(z,3) for z in y]
        
 curRow=X[2000]

# Original Prediction
print("Original Prediction")
print(rfModel.predict_proba([curRow]))

def getPrediction(curRow,tree1,tree2):
    probValues=[]
    for x in range(len(tree1)):
        children_left=tree1[x][:,0]
        children_right=tree1[x][:,1]
        feature=tree1[x][:,2]
        threshold=tree2[x][:,0]
        values0=tree2[x][:,1]
        values1=tree2[x][:,2]
        sumVal=values0+values1
        curNode=0
        while(children_left[curNode] != -1 or children_right[curNode] != -1):
            if(curRow[int(feature[curNode])] <= threshold[curNode]):
                curNode=int(children_left[curNode])
            else:
                curNode=int(children_right[curNode])
        probValues.append([values0[curNode]/sumVal[curNode],values1[curNode]/sumVal[curNode]])
        
        #print(probValues)
    return(np.mean(probValues,axis=0))
        
print("Calculated Prediction")
print(getPrediction(curRow,tree1,tree2))

pickle.dump(tree1_1,open("treeModel1","wb") )
pickle.dump(tree2_1,open("treeModel2","wb") )
sizeVal=os.stat('treeModel1').st_size + os.stat('treeModel2').st_size
print("Size of treeModel is {0}".format(sizeVal/(1024*1024)))
np.save("numpyModel1",np.array(tree1_1))
np.save("numpyModel2",np.array(tree2_1))
sizeVal=os.stat('numpyModel1.npy').st_size + os.stat('numpyModel2.npy').st_size
print("Size of treeModel is {0}".format(sizeVal/(1024*1024)))
