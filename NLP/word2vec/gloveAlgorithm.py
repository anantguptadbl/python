# GLOVE Algorithm on Movie Plots

# READ DATA
import pandas as pd
data=pd.read_csv('wiki_movie_plots_deduped.csv.tar.gz')
data=data.head(1000)

# GLOVE Dataset Algorithm
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import string
from itertools import groupby
import math

# Global List
stopWords = set(stopwords.words('english'))
# Take only those words that are noun, verbs, adjectives and adverbs
posTagList=['NN','NNS','NNP','NNPS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS']
# Special character removal
specialChars='(),#-:"\/{}[]_@'
multipleRemovals=['\r','\n','\xe2','\x80','\x94',"\'"]
printable = set(string.printable)
lemmatizer = WordNetLemmatizer()

# Data PreProcessing
def removeStopWords(x):
    return(' '.join([y for y in str(x).split(' ') if str(y).lower() not in stopWords and len(str(y).strip()) > 0]))

def getValidPosTag(x,posTagList):
    x1=str(x).lower()
    curTag=nltk.pos_tag([x1])
    if(curTag[0][1] in  posTagList) :
        return x
    else:
        return ''

def selectPosTags(x,posTagList):
    return(' '.join([getValidPosTag(y,posTagList) for y in str(x).split(' ') if len(str(y).strip()) > 0]))

def removeSpecialChars(x,specialChars):
    for y in specialChars:
        x=x.replace(y,'')
    return x

def removeMultipleChars(x,multipleRemovals):
    for y in multipleRemovals:
        x=x.replace(y,'')
    return(x)

def removenonAscii(x):
    return(filter(lambda y: y in printable, x))

def lemmatize(x):
    return([lemmatizer.lemmatize(z) for y in x.split('.') for z in y.split(' ')])
    
    
data['Plot']=data['Plot'].map(lambda x : removeStopWords(x))
data['Plot']=data['Plot'].map(lambda x : selectPosTags(x,posTagList))
print("Filter of pos tags completed")
data['Plot']=data['Plot'].map(lambda x : removeSpecialChars(x,specialChars))
print("Removal of specialchars completed")
data['Plot']=data['Plot'].map(lambda x : removeMultipleChars(x,multipleRemovals))
print("Removal of multiple chars completed")
data['Plot']=data['Plot'].map(lambda x : re.sub(r'\[.+\]', '', x, re.I))
data['Plot']=data['Plot'].map(lambda x : re.sub(r'x\w\w', '', x, re.I))
print("Removal of regex completed")
data['Plot']=data['Plot'].map(lambda x : removenonAscii(x))
print("Removal of non ascii completed")
data['Plot']=data['Plot'].map(lambda x : lemmatizer.lemmatize(x))
print("All operations completed")

# Find the count of the word and context as per 1 skipgram
FullCount=[]
def getCount(x,FullCount):
    # First we will split into individual sentences
    for y in x.split('.'):
        z=y.split(' ')
        z=[str(v).lower() for v in z if len(v) > 1]
        for w in range(len(z) - 2):
            FullCount.append([z[w],z[w+1]])
            FullCount.append([z[w],z[w+2]])
        
data['Plot'].map(lambda x : getCount(x,FullCount))
FullCount=[ k+[len(list(g))] for k,g in groupby(FullCount)]
FullCount=pd.DataFrame(FullCount,columns=['word1','word2','count'])
FullCount['countLog']=FullCount['count'].map(lambda x : math.log(x))

# Get all the words
FullWords=list(set(list(FullCount['word1'].unique()) + list(FullCount['word2'].unique())))

# We will now be perfoming the SGD algorithm
