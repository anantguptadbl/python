import pandas as pd
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.utils import lemmatize
# We will now model using simple ANN for now
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


# KAGGLE DATA SOURCE
# https://www.kaggle.com/jrobischon/wikipedia-movie-plots
wikiData=pd.read_csv("wikipedia-movie-plots/wiki_movie_plots_deduped.csv")

# Filling Missing Genre
np.sum(wikiData['Genre']=='unknown')

# Out of 34886 movies 6083 have genre missing

import re
pattern = '^[^&0123456789-].*$'

genres=wikiData['Genre'].unique()
genres=[x.split(',') for x in genres]
genres=[y.split(' ') for x in genres for y in x ]
genres=[y.split('/') for x in genres for y in x]
genres=[str(x[0]) for x in genres if len(str(x[0])) > 0]
genres=set(genres)
genres=[ y[0] for y in [re.match(pattern, x) for x in genres] if y is not None]
genres=[x for x in genres if x[0] != '(']
genres=[x for x in genres if len(x) > 1]

for curGenre in genres:
    wikiData['curGenre']=0
    
for curGenre in genres:
    #print("CurGenre is {0}".format(curGenre))
    wikiData[curGenre]=wikiData['Genre'].apply(lambda x : 1 if curGenre in x else 0)
    
# Filtering out unnecessary genres
genreCount=np.sum(wikiData[genres],axis=0)
discardGenres=genreCount[genreCount<=5].index.values
genres=[x for x in genres if x not in discardGenres]

# Deleting the genres
del(wikiData['curGenre'])
for curGenre in discardGenres:
    del(wikiData[curGenre])
    
# Train Test Split
wikiData['trainTest']=wikiData.apply(lambda row : 'train' if np.sum(row[genres]) > 0 else 'test' ,axis=1)

def lemmaSentence(curSentence):
    x=lemmatize(curSentence)
    x=set([y.decode('utf-8').split('/')[0] for y in x])
    x=[str(y).lower() for y in x if len(y) > 2]
    #print("Completed")
    return(x)

def lemmaSentence1(i,curSentence):
    x=lemmatize(curSentence)
    x=set([y.decode('utf-8').split('/')[0] for y in x])
    x=[str(y).lower() for y in x if len(y) > 2]
    #print("Completed")
    print("Completed for i {0}".format(i))
    return(TaggedDocument(words=x, tags=[str(i)]))
    
print("Tagged_data created")

model=Doc2Vec.load("GensimModelDoc2Vec_moviePlot")

for epoch in [500]:
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,total_examples=model.corpus_count,epochs=epoch)
    model.alpha -= 0.01
    model.min_alpha = model.alpha

# Save the model
model.save("GensimModelDoc2Vec_moviePlot")

class predictGenre(nn.Module):
    def __init__(self):
        super(predictGenre,self).__init__()
        self.l1=nn.Linear(400,200)
        #self.l2=nn.Linear(300,200)
        self.l3=nn.Linear(200,164)
        self.sig=nn.Sigmoid()
        self.r1=nn.LeakyReLU(0.1)
        self.r2=nn.LeakyReLU(0.1)
        
    def forward(self,input):
        out=self.l1(input)
        out=self.r1(out)
        #out=self.l2(out)
        #out=self.r2(out)
        out=self.l3(out)
        out=self.sig(out)
        return(out)
    
torch.manual_seed(42)
genreModel = predictGenre()
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

print("Starting to extract the vector")
# Extracting the vector
wikiData.loc[0:30000,'doc2vec']=pd.Series([[round(y,5) for y in model.infer_vector(word_tokenize(x))] for x in wikiData['Plot'].values[0:30000]])
print("Vectors have been extracted")

# Setting up the training data
trainData=np.array([np.array(x).reshape(400) for x in wikiData[0:30000][wikiData['trainTest']=='train']['doc2vec'].values])
trainLabels=np.array(wikiData[0:30000][wikiData['trainTest']=='train'][genres].values[0:30000])
testData=np.array([np.array(x).reshape(400) for x in wikiData[0:30000][wikiData['trainTest']=='test']['doc2vec'].values])

criterion = nn.BCELoss()
#optimizer = torch.optim.Adam(genreModel.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = torch.optim.Adam(genreModel.parameters(), lr=0.0001)
epochSize=10000
batchSize=1024
miniBatches=29
for curEpoch in range(epochSize):
    curLoss=0
    for curBatch in range(miniBatches):
        genreModel.zero_grad()
        #choiceList=np.random.choice(34845,batchSize)
        #train=Variable(torch.from_numpy(trainData[choiceList]))
        #labels=Variable(torch.from_numpy(trainLabels[choiceList].astype(np.float32)))
        train=Variable(torch.from_numpy(trainData[curBatch*batchSize:(curBatch+1)*batchSize]))
        labels=Variable(torch.from_numpy(trainLabels[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32)))
        output=genreModel(train)
        #print("Size of train and labels and output is {0} and {1} and {2}".format(train.size(),labels.size(),output.size()))
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        curLoss=curLoss + loss.item()
    if(curEpoch % 10==0):
        print("Epoch {0} TotalLoss : {1}".format(curEpoch,curLoss))
        
torch.save(genreModel,"genreModel_20191009")

# Testing
pd.DataFrame(list(zip(genres,[round(x,2) for x in genreModel(Variable(torch.from_numpy(np.array(wikiData['doc2vec'].values[56]).reshape(1,-1)))).detach().numpy()[0]])),columns=['genre','score']).sort_values(by=['score'],ascending=False)




