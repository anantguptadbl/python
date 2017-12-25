import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
os.chdir("KaggleBagOfwords/")

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0,delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0,  delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " "and %d unlabeled reviews\n" % (train["review"].size, test["review"].size,unlabeled_train["review"].size )

def dataCleansingword2vec(rawText,remove_stopwords=False):
	# Removing tags
	reviewText=BeautifulSoup(rawText).get_text()
	# Taking only alpha
	letters_only = re.sub("[^a-zA-Z]", " ", reviewText)
	# Split sentence to words
	words = letters_only.lower().split()
	# Exclude stopwords
	#stops = set(stopwords.words("english"))
	#meaningful_words = [w for w in words if not w in stops]
	# Seam it back	
	return( " ".join( words )) 

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    #review=review.decode('utf-8').encode('utf-8')	
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( dataCleansingword2vec( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
sentences += review_to_sentences(review, tokenizer)
