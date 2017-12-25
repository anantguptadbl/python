# Kaggle Michigan

# Read the data
import nltk
from nltk import FreqDist
import pandas as pd
mich=pd.read_csv("/home/anantgupta/Documents/Python/MachineLearning/KaggleMichigan/training.txt",sep='\t')
mich.columns=['Val','Text']

# After we have set the data frame we will create a feature extractor

# LOGIC 1
# We will remove the stopwords and then include the contains logic
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) 
# It is not necessary to remove the stop words in the document itself

# In the feature we will be including words that have been used the most in the entire text
fullText=''
for sentence in mich['Text']:
	fullText=fullText + ' ' + sentence

highestWords = FreqDist(w.lower() for w in fullText.split(' ') if w not in stop_words and len(w) > 2)
highestWordsTop1000=list(highestWords)[:1000]

def document_features(sentence):
	document_words = sentence.split(' ')
	features = {}
	for word in highestWordsTop1000:
        	features['contains({})'.format(word)] = (word in document_words)
	return features

featuresets = [(document_features(row['Text']), row['Val']) for index,row in mich.iterrows() if len(row['Text'])>0]
features = [(document_features(row['Text'])) for index,row in mich.iterrows() if len(row['Text'])>0]
train_set, test_set = featuresets[5000:], featuresets[:1917]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

print(nltk.classify.accuracy(classifier, test_set))
# 0.9848. this means our classifier is good

# Let us now use the classifier on the test data
michTest=pd.read_csv("/home/anantgupta/Documents/Python/MachineLearning/KaggleMichigan/testdata.txt",sep='\t')
features = [(document_features(row['Text'])) for index,row in michTest.iterrows() if len(row['Text'])>0]

def classifyResults(sentence):
	return classifier.classify(document_features(sentence))

michTest['Val']=map(classifyResults,michTest['Text'])
