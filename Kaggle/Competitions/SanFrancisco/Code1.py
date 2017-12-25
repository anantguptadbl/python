# Kaggle San Francisco submission

import pandas as pd
import math
train=pd.read_csv('KaggleSanFrancisco/train.csv')
test=pd.read_csv('KaggleSanFrancisco/test.csv')
submission=pd.read_csv('KaggleSanFrancisco/sampleSubmission.csv')
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
plotData=train[['DayOfWeek','Category']].groupby(['DayOfWeek','Category']).size()
plotDataDF = pd.DataFrame({'count' : train[['DayOfWeek','Category']].groupby(['DayOfWeek','Category']).size()}).reset_index()

train['XAbs'] = map(math.floor,train['X'])
train['YAbs'] = map(math.floor,train['Y'])
train['XVar'] = map(lambda x: x - math.floor(x),train['X'])
train['YVar'] = map(lambda x: x - math.floor(x),train['Y'])
S = set(train['PdDistrict']) # collect unique label names
D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
train['district1'] = [D[y2_] for y2_ in train['PdDistrict']]

S = set(train['DayOfWeek']) # collect unique label names
D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
train['day1'] = [D[y2_] for y2_ in train['DayOfWeek']]

##########################
# TEST DATA
##########################

test['XAbs'] = map(math.floor,test['X'])
test['YAbs'] = map(math.floor,test['Y'])
test['XVar'] = map(lambda x: x - math.floor(x),test['X'])
test['YVar'] = map(lambda x: x - math.floor(x),test['Y'])
S = set(test['PdDistrict']) # collect unique label names
D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
test['district1'] = [D[y2_] for y2_ in test['PdDistrict']]

S = set(train['DayOfWeek']) # collect unique label names
D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
test['day1'] = [D[y2_] for y2_ in test['DayOfWeek']]


for district in train['PdDistrict'].unique():
	print(district)
	trainSubSet = train[train['PdDistrict']==district]
	trainSubSet.groupby('PdDistrict').count()
	train1, test1 = train_test_split(trainSubSet, train_size=30000)
	from sklearn.cross_validation import cross_val_score
	from sklearn.datasets import make_blobs
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.tree import DecisionTreeClassifier
	forest = RandomForestClassifier(n_estimators = 50,max_depth=10,bootstrap=False,verbose=1)
	forest = forest.fit(train1[['district1','day1','XAbs','YAbs','XVar','YVar']],train1['Category'])
	# We will have to loop through day1
	for dayunique in test.loc[test['PdDistrict']==district,'day1'].unique():
		test.loc[(test['PdDistrict']==district) & (test['day1']==dayunique),'status']=forest.predict(test.loc[(test['PdDistrict']==district) & (test['day1']==dayunique),['district1','day1','XAbs','YAbs','XVar','YVar']])

# Now we will prepare the submission csv file
test['count']=1
resultFile = test.pivot_table(values='count',index='Id',columns='status',aggfunc=np.sum)

# Check for columns which are not there in the result file and there in the sampleSubmission File
for columnName in sampleSubmission.columns.values:
	if columnName in resultFile.columns.values==False:
		resultFile[columnName]=0

# The final file is ready

#from sklearn.metrics import accuracy_score
#accuracy_score(test1['Category'],test1['status'],normalize=True)
