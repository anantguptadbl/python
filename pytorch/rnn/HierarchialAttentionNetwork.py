#import pandas as pd
#data=pd.read_csv('hotel-review/train.csv')

import nltk

validPosTags=['JJ','JJR','JJS','NN','RB','RBR','RBS','VB','VBD','VBG','VBZ']

def removeUnncessaryTags(x):
    # For ech sentnece in a row
    # For each word in a sentence
    validSentences=[]
    for curSentence in x.split('.'):
        validWords=[]
        for curWord in curSentence.split(' '):
            if(len(curWord.strip()) > 2):
                if(nltk.pos_tag([curWord.lower()])[0][1] in validPosTags):
                    validWords.append(curWord.lower())
        validSentences.append(validWords)
    return(validSentences)
            
                
data=data[0:10000]
data['data']=data['Description'].map(lambda x : removeUnncessaryTags(x))
data.head(5)
