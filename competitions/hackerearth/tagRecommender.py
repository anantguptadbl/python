import pandas as pd
import numpy as np
import scipy

data=pd.read_csv('train.csv',sep=',')
removalChars=[x for x in ':[]{}?-_"()/\\']
def charRemoval(word):
    word=re.sub(r'[^\x00-\x7F]+','',word)
    return(''.join([x for x in word if x not in removalChars]))

data['title']=data['title'].map(lambda x : charRemoval(x))

import nltk
import re
from nltk.stem import WordNetLemmatizer

# First we will work only with title and tags

# We will remove all adjectives, pronouns etc from the title
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer=nltk.PorterStemmer()

# Tag Removal
#removalTags=['CC','CD','DT','EX','PRP','PRP$','TO','UH','SYM','UH','WP','IN','WRB',':']
# Excluding Verbs
removalTags=['CC','CD','DT','EX','PRP','PRP$','TO','UH','SYM','UH','WP','IN','WRB',':','VB','VBG','VBP','VBD','VBN','VBZ']

    
def wordLemma(word):
    return(porter_stemmer.stem(wordnet_lemmatizer.lemmatize(wordnet_lemmatizer.lemmatize(word,'v'),'n')))

def checkPosTag(word):
    numbers=[str(x) for x in range(10)]
    if any([True if x in word else False for x in numbers]):
        return False
    try:
        if(float(word.strip())):
            return False
    except:
        if(len(word.strip())==0):
            return False
        posTag=nltk.pos_tag([word])
        if len(posTag) > 0:
            if posTag[0][1] not in removalTags:
                return True
    return False

titleTags=[]
for index,curRow in data[0:100000].iterrows():
    if(index % 2000==0):
        print("Completed for {}".format(index))
    titleTags.append([
        curRow[0],
        [wordLemma(y.strip().lower()) for y in str(curRow[1]).split(' ') if checkPosTag(y)==True and len(str(y)) > 1]
    ]
    )
    
# Working with only 100K records for now
titleTags=pd.DataFrame(titleTags,columns=['id','titleTag'])
titleTags=titleTags.merge(data[['id','tags']],left_on='id',right_on='id',how='inner')
titleTags['tags']=titleTags['tags'].map(lambda x : str(x).split('|'))

# CASE 1: Using graphs
# We will do this using networkx

import networkx as nx
g=nx.Graph()
for curIndex,curRow in titleTags.iterrows():
    for curSource in curRow['titleTag']:
        for curDest in curRow['tags']:
            if(curSource not in g):
                g.add_node('source_' + str(curSource))
            if(curDest not in g):
                g.add_node('dest_' + str(curDest))
            if g.has_edge('source_' + str(curSource), 'dest_' + str(curDest)):
                g['source_' + str(curSource)]['dest_' + str(curDest)]['weight'] += 1
            else:
                g.add_edge('source_' + str(curSource),'dest_' + str(curDest), weight=1)
                
def getPrediction(inpString):
    tempResults=[]
    for x in g['source_' + str(inpString)]:
        tempResults.append([x,g['source_' + str(inpString)][x]['weight']])
    return(pd.DataFrame(tempResults,columns=['dest','weight']).sort_values('weight',ascending=False).head(5)[['dest','weight']].values)
    
# CASE 2 : Passing TFIDF output to neural net
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [' '.join(x) for x in titleTags['titleTag']]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.shape)

import tensorflow as tf
class TFAutoEncoder():
    # INIT function
    def __init__(self,X,Y,learningRate):
        # Input Data
        self.X=X
        self.Y=Y
        
        # Input Placeholder
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        self.y_input=tf.placeholder("float32",(None,Y.shape[1]))
        #self.x_input=tf.sparse_placeholder("float32",(None,X.shape[1]))
        
        # Intermediate Variables
        self.encoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1],self.X.shape[1]/2]))
        self.encoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]/3]))
        self.decoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/3,self.Y.shape[1]/2]))
        self.decoder_2_weight=tf.Variable(tf.random_uniform([self.Y.shape[1]/2,self.Y.shape[1]]))
        self.encoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.encoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/3]))
        self.decoder_1_bias=tf.Variable(tf.random_uniform([self.Y.shape[1]/2]))
        self.decoder_2_bias=tf.Variable(tf.random_uniform([self.Y.shape[1]]))
                
        #self.encoder_1=tf.add(tf.sparse_tensor_dense_matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias)
        self.encoder_1=tf.add(tf.matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias)
        self.encoder_2=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_1,self.encoder_2_weight),self.encoder_2_bias))
        self.decoder_1=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_2,self.decoder_1_weight),self.decoder_1_bias))
        self.decoder_2=tf.add(tf.matmul(self.decoder_1,self.decoder_2_weight),self.decoder_2_bias)
        
        #self.loss=tf.sparse_reduce_sum(tf.sparse_add(self.x_input,-self.decoder_2))
        #self.loss=tf.reduce_mean(tf.sparse_add(self.x_input,-self.decoder_2))
        #self.loss=tf.reduce_mean(tf.pow(self.x_input-self.decoder_2,2))
        #self.loss=tf.reduce_mean(tf.pow(self.y_input-self.decoder_2,2))
        self.loss=tf.losses.softmax_cross_entropy(self.y_input,self.decoder_2)
        self.optimizer=tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)
        self.init=tf.global_variables_initializer()
        
    def train(self,execRange=1000):
        with tf.Session() as sess:
            sess.run(self.init)
            for curIteration in range(execRange):
                _,curLoss=sess.run([self.optimizer,self.loss],feed_dict={self.x_input:self.X,self.y_input:self.Y})
                if(curIteration % 100==0):
                    print("The loss at step {} is {}".format(curIteration,curLoss))
                
#data=scipy.sparse.csr_matrix(np.random.rand(1000,5))
data=np.random.rand(1000,5)
data1=np.random.rand(1000,4)
ae=TFAutoEncoder(data,data1,0.01)
ae.train(5000)
