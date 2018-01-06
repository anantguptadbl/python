import urllib2
import pandas as pd
import numpy as np
import requests
requests.adapters.DEFAULT_RETRIES = 1
from bs4 import BeautifulSoup
import copy
import json
import re
import time
import re
import nltk
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tensorflow as tf


def preProcessing(data):
    # Remove Special Characters
    data=re.sub(r'[^\x00-\x7f]+',' ',data)
    # Simple Pre Processing
    data=data.replace('\n','').replace('\r','').replace('\t','').replace('(','').replace(')','')
    # Data within double quotes have to proper cased
    matches = re.findall(r'\"(.+?)\"',data)  # match text between two quotes
    for m in matches:
        data = data.replace('"%s"'%m,m.title())  # override text to include tags
    return(data)

def combineProperNouns(a):
    y=0
    while y <= len(a)-2:
        if(a[y][0].isupper()==True and a[y+1][0].isupper()==True):
            a[y]=str(a[y]) + '+' + str(a[y+1])
            a[y+1:]=a[y+2:]
        else:
            y=y+1
    return(a)

def recreateDataWithCombinedProperNouns(data):
    tempData=[]
    for x in data.split('.'):
        tempPhrase=[]
        for y in x.split(','):
            z=y.split(' ')
            z=[a for a in z if len(a) > 0]
            tempPhrase.append(' '.join(combineProperNouns(z)))
        tempData.append(','.join(tempPhrase))
    data='.'.join(tempData)
    return(data)

def removeDotsFromAcronyms(data):
    counter=0
    while counter < len(data) -2:
        if(data[counter]=='.' and data[counter+2]=='.'):
            #print("######{}#####{}#######{}####".format(counter,data[counter-1:counter+3],data[counter+1]))
            data=data[:counter] + str(data[counter+1]) + ' ' + data[counter+3:]
            counter=counter+1
        elif(data[counter]=='.' and data[counter-1].isupper()==True):
            #print("####{}####".format(data[counter-1:counter+1]))
            data=data[:counter] + data[counter+1:]
        else:
            counter=counter+1
    return(data)


class wordSimilarity():
    uniqueWords=[]
    fullString=''
    sentences=[]
    vocab_size=0
    words=[]
    # Initialization Function
    def __init__(self,data):
        self.fullString=data    
   
    # Get uniqueWords
    def __setUniqueWords(self):
        self.uniqueWords=[]
        for sentence in self.fullString.split('.'):
            self.uniqueWords=self.uniqueWords + sentence.split(' ')
        self.uniqueWords=list(set(self.uniqueWords))
        self.vocab_size=len(self.uniqueWords)
    
    # Get fullString
    def _getFullString(self):
        for sentence in self.sentences:
            self.fullString=self.fullString + sentence
    
    #Split into sentences
    def __getSentences(self):
        self.sentences=self.fullString.split('.')
        
    #Stemming and Lemming
    def stemAndLem(self):
        self.sentences=[stemAndLemmatize(x) for x in self.sentences]
        self.fullString='.'.join(self.sentences)
        
    # Filter by posTagList
    def preProcessData(self):
        self.fullString=preProcessing(self.fullString)
        self.fullString=recreateDataWithCombinedProperNouns(self.fullString)
        self.fullString=removeDotsFromAcronyms(self.fullString)
        self.__getSentences()
         
    def removeUnnecessaryCharacter(self):
        self.fullString=self.fullString.replace(',',' ').replace('?','').replace(';',' ').replace("'",'')
        self.sentences=self.fullString.split('.')
        
    def removeTags(self,tagList,sentence):
        return(' '.join([ y for y in sentence.split(' ') if '+' in y or (len(nltk.pos_tag(y.split()))>0 and nltk.pos_tag(y.split())[0][1] in tagList)]))
    
    def removeTagSentences(self,tagList):
        self.sentences=[self.removeTags(tagList,x) for x in self.sentences]
        self.fullString='.'.join(self.sentences)
        
    def getDictsForWord2Vec(self):
        self.word2int={}
        self.int2word={}
        self.__setUniqueWords()
        for i,word in enumerate(self.uniqueWords):
            self.word2int[word]=i
            self.int2word[word]=i
            
    def generateSkipWords(self,sentence):
        for word_index,word in enumerate(sentence):
            for nb_word in sentence[max(word_index - self.WINDOW_SIZE, 0) : min(word_index + self.WINDOW_SIZE, len(sentence)) + 1]: 
                if nb_word != word:
                    self.words.append([word,nb_word])
    
    def runSkipWords(self,WINDOW_SIZE):
        self.WINDOW_SIZE=WINDOW_SIZE
        [self.generateSkipWords(x1.split(' ')) for x1 in self.sentences]
                
    def to_one_hot(self,data_point_index, vocab_size):
        temp = np.zeros(vocab_size)
        temp[data_point_index] = 1
        return temp

    def createTrainTestData(self):
        self.x_train = [] # input word
        self.y_train = [] # output word
        for data_word in self.words:
            self.x_train.append(self.to_one_hot(self.word2int[ data_word[0] ], self.vocab_size))
            self.y_train.append(self.to_one_hot(self.word2int[ data_word[1] ], self.vocab_size))
        # convert them to numpy arrays
        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)
        
    def tensorFlowInitialization(self,EMBEDDING_DIM):
        self.x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        self.y_label = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        self.EMBEDDING_DIM = EMBEDDING_DIM # you can choose your own number
        self.W1 = tf.Variable(tf.random_normal([self.vocab_size, self.EMBEDDING_DIM]))
        self.b1 = tf.Variable(tf.random_normal([self.EMBEDDING_DIM])) #bias
        self.hidden_representation = tf.add(tf.matmul(self.x,self.W1), self.b1)
        self.W2 = tf.Variable(tf.random_normal([self.EMBEDDING_DIM, self.vocab_size]))
        self.b2 = tf.Variable(tf.random_normal([self.vocab_size]))
        self.prediction = tf.nn.softmax(tf.add( tf.matmul(self.hidden_representation, self.W2), self.b2))    
    
    def modelTraining(self,n_iters=10):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init) #make sure you do this!
        
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(self.y_label * tf.log(self.prediction), reduction_indices=[1]))
        # define the training step:
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
        print("We will start training now")
        # train for n_iter iterations
        for _ in range(n_iters):
            sess.run(train_step, feed_dict={self.x: self.x_train, self.y_label: self.y_train})
            print('loss is : ', sess.run(cross_entropy_loss, feed_dict={self.x: self.x_train, self.y_label: self.y_train}))
        
        self.vectors=sess.run(self.W1 + self.b1)
