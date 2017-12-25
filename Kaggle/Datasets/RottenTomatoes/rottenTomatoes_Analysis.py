# Import Libraries
import pandas as pd
import numpy as np
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

data=pd.read_csv("KaggleRottenTomatoes/train.tsv",sep="\t")
data['Count']=[len(x.split(' ')) for x in data['Phrase']]

# Now we will choose the phraseId or which the count is the max
#newData=data.loc[data.loc[data['SentenceId']==x,['Count']].idxmax()] for x in data['SentenceId'].unique()
#newData1=pd.DataFrame()
#for r in newData:
#	newData1=newData1.append(r)
newData1=data

# Cleaning up the stop words from each phrase
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

newData1['NLTK']=[[word for word in newData1['Phrase'].iloc[i].lower().split(' ') if word not in stop_words] for i in xrange(0,len(newData1))]

# We will be trying with a simple features on the newly created DataFrame
def word_feats(words):
    return dict([(word, True) for word in words])

feats=[(word_feats(newData1['NLTK'].iloc[i]),newData1['Sentiment'].iloc[i]) for i in xrange(0,len(newData1))]
#classifier = NaiveBayesClassifier.train(data)
import nltk
classifier = nltk.NaiveBayesClassifier.train(feats)

# We will now try it on the test data
test=pd.read_csv("KaggleRottenTomatoes/test.tsv",sep="\t")
test['Count']=[len(x.split(' ')) for x in test['Phrase']]

# Now we will choose the phraseId or which the count is the max
#newTest=[test.loc[test.loc[test['SentenceId']==x,['Count']].idxmax()] for x in test['SentenceId'].unique()]
#newTest1=pd.DataFrame()
#for r in newTest:
#	newTest1=newTest1.append(r)
newTest1=test

newTest1['NLTK']=[[word for word in newTest1['Phrase'].iloc[i].lower().split(' ') if word not in stop_words] for i in xrange(0,len(newTest1))]
featsTest=[(word_feats(newTest1['NLTK'].iloc[i])) for i in xrange(0,len(newTest1))]
newTest1['Sentiment']=[classifier.classify(x) for x in featsTest]
finalResult=newTest1.loc[:,['PhraseId','Sentiment']]
finalResult.to_csv('KaggleRottenTomatoes/submissionFile.csv',index=False)
