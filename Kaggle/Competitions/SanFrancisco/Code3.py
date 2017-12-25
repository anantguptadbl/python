# Kaggle San Francisco submission

import pandas as pd
import math
import numpy as np
from __future__ import division
train=pd.read_csv('KaggleSanFrancisco/train.csv')
test=pd.read_csv('KaggleSanFrancisco/test.csv')
submission=pd.read_csv('/home/anantgupta/Documents/Python/MachineLearning/KaggleSanFrancisco/sampleSubmission.csv')
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import matplotlib.pyplot as plt
plotData=train[['DayOfWeek','Category']].groupby(['DayOfWeek','Category']).size()
plotDataDF = pd.DataFrame({'count' : train[['DayOfWeek','Category']].groupby(['DayOfWeek','Category']).size()}).reset_index()

# Function to convert X Y coordinates to small boxes
def XYtoBoxes(X,Y,XStart,YStart,XEnd,YEnd,incrementVal):
	outputVal=[]
	YCount=round((YEnd - YStart)/incrementVal)
	print(YCount)
	for indexVal in xrange(0,len(X)):
		diff1=X[indexVal]-XStart
		#print(diff1)
		diff2=Y[indexVal]-YStart
		#print(diff2)
		outputVal.append(round(diff1/incrementVal) + (round(diff2/incrementVal) * YCount ))
	return(outputVal)
# End of the function

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
train['Dates1'] = pd.to_datetime(train['Dates'])
train['Hour'] = pd.DatetimeIndex(train['Dates1']).hour
train['XYCube']=XYtoBoxes(train['X'],train['Y'],-122,37,-120,40,0.05)

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
test['Dates1'] = pd.to_datetime(test['Dates'])
test['Hour'] = pd.DatetimeIndex(test['Dates1']).hour
test['XYCube']=XYtoBoxes(test['X'],test['Y'],-122,37,-120,40,0.05)

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
	#forest = RandomForestClassifier(n_estimators = 100,max_depth=10,bootstrap=False)
	#forest = forest.fit(train1[['district1','day1','XYCube','Hour']],train1['Category'])
	#test1['evaluated']=forest.predict(test1[['district1','day1','XYCube','Hour']])
	#correct=sum(test1['evaluated']==test1['Category'])
	#print("The ratio is " + str(correct/len(test1)) )
	# Multinomial NB
	#clf = MultinomialNB()	
	#clf.fit(train1[['day1','XYCube','Hour']],train1['Category'])
	#test1['evaluated']=clf.predict(test1[['day1','XYCube','Hour']])
	#correct=sum(test1['evaluated']==test1['Category'])
	#print("The MNB ratio is " + str(correct/len(test1)) )
	# Gaussian NB	
	#clf = GaussianNB()
	#clf.fit(train1[['day1','XYCube','Hour']],train1['Category'])
	#test1['evaluated']=clf.predict(test1[['day1','XYCube','Hour']])
	#correct=sum(test1['evaluated']==test1['Category'])
	#print("The GB ratio is " + str(correct/len(test1)) )
	# SVM
	clf = svm.SVC()
	clf.fit(train1[['day1','XYCube','Hour']],train1['Category'])
	test1['evaluated']=clf.predict(test1[['day1','XYCube','Hour']])
	correct=sum(test1['evaluated']==test1['Category'])
	print("The SVM ratio is " + str(correct/len(test1)) )
	#print("Score is " + str(score(forest,test1['Category'], forest.predict(test1[['district1','day1','XYCube','Hour']])))
	# We will have to loop through day1
	for dayunique in test.loc[test['PdDistrict']==district,'day1'].unique():
		test.loc[(test['PdDistrict']==district) & (test['day1']==dayunique),'status']=clf.predict(test.loc[(test['PdDistrict']==district) & (test['day1']==dayunique),['day1','XYCube','Hour']])

# Now we will prepare the submission csv file
test['count']=1
resultFile = test.pivot_table(values='count',index='Id',columns='status',aggfunc=np.sum)
resultFile=resultFile.fillna(0)

# Check for columns which are not there in the result file and there in the sampleSubmission File
for columnName in submission.columns.values:
	if (columnName in resultFile.columns.values)==False:
		resultFile[columnName]=0

resultFile['Id']=resultFile.index.values

# The final file is ready
resultFile.to_csv('KaggleSanFrancisco/results_2.csv',index=False)
#from sklearn.metrics import accuracy_score
#accuracy_score(test1['Category'],test1['status'],normalize=True)
