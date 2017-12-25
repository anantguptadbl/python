import sqlite3
#import csv
import unicodecsv as csv
import codecs
#content = unicode(q.content.strip(codecs.BOM_UTF8), 'utf-8')
#parser.parse(StringIO.StringIO(content))

conn=sqlite3.connect('/home/anantgupta/Documents/Python/MachineLearning/KaggleClinton/output/database.sqlite')
cur = conn.cursor()
cur.execute("select name from sqlite_master where type='table'")
print cur.fetchone()

# We see that the table name is Emails
cur.execute("select count(*) from Emails")
print cur.fetchone()

# We see that the number of emails are 7945
cur.execute("select * from Emails")
with open("/home/anantgupta/Documents/Python/MachineLearning/KaggleClinton/output/emailData.csv", "wb") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in cur.description]) # write headers
    csv_writer.writerows(cur)

###### 2016-04-30

import networkx as nx
import re
import os
import pandas as pd
import numpy as np
os.chdir("/home/anantgupta/Documents/Python/MachineLearning/KaggleClinton/")
df_aliases = pd.read_csv('./output/Aliases.csv', index_col=0)
df_emails = pd.read_csv('./output/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('./output/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('./output/Persons.csv', index_col=0)


emaildata=df_emails

# Creating GRAPHS for emails sent
# First we will have to cleanse the MetadataTo column

# Make the entire data lower case
emaildata.MetadataFrom=emaildata.MetadataFrom.str.lower()
mask = emaildata['MetadataFrom'].str.len() > 2
emaildata = emaildata.loc[mask]
emaildata=emaildata.dropna(subset=['MetadataFrom'])

# Now we will have to split the data by ; and remove the portion having length < 2
def getValidRecipient(x):
	print(x)
	y=x.split(';')
	for recipient in y:
		if len(str(recipient)) > 2:
			return cleanRecipient(recipient)

def cleanRecipient(x):
	print("Before cleaning " + x)
	# Remove text after the @ sign
	x = re.sub(r"@.*$", "", x)
	
	# Also we need to remove single characters in the names of the people
	x=re.sub(r"\s[a-zA-Z0-9]{1}\s"," ",x)
	x=re.sub(r"\s[a-zA-Z0-9]{1}$","",x)
	x=re.sub(r"^[a-zA-Z0-9]{1}\s","",x)
	
	# We need to replace the , with space for uniformity
	x=re.sub(r","," ",x)
	x=re.sub(r"\s+"," ",x)
	
	# Returning the final value. We can further improve on this function
	print("After cleaning " + x)
	return x

emaildata['fromaddress']=map(getValidRecipient,emaildata.MetadataFrom)



SentGraph = pd.pivot_table(emaildata, index=['fromaddress'], aggfunc=np.sum)
SentGraphTop10 = SentGraph.sort_values(by='SenderPersonId')
SentGraphTop10 = SentGraphTop10.tail(10)
# Create a node for Hillary
EmailReceived = nx.DiGraph()
EmailReceived.add_node("Hillary",label="hillary")
#for x in emaildata['fromaddress'].unique():
for x in SentGraphTop10.index.values:
	#EmailReceived.add_node(x)
	print(x)
	#EmailReceived.add_weighted_edges_from(("Hillary",str(x),SentGraph.loc[x][0]))
	#EmailReceived.add_weighted_edges_from(("Hillary",str(x)))
	EmailReceived.add_edge("Hillary",x,label=x)

graphObject=nx.shell_layout(EmailReceived)
nx.draw_networkx_nodes(EmailReceived,graphObject)
nx.draw_networkx_edges(EmailReceived,graphObject)
nx.draw_networkx_labels(EmailReceived, graphObject)
plt.show()


# Creating GRAPHS for emails received


###### End of 2016-04-30

###### 2016-05-01

# We will now be doing direct sentiment Analysis
import networkx as nx
import re
import os
import pandas as pd
import numpy as np
import json
os.chdir("/home/anantgupta/Documents/Python/MachineLearning/KaggleClinton/")
df_aliases = pd.read_csv('./output/Aliases.csv', index_col=0)
df_emails = pd.read_csv('./output/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('./output/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('./output/Persons.csv', index_col=0)

from nltk.stem.porter import PorterStemmer
import nltk

os.chdir("/home/anantgupta/stanford-corenlp-python/")

from corenlp import *
corenlp = StanfordCoreNLP()  # wait a few minutes...
#result=corenlp.parse("Parse this sentence.")

def isLocation(sentenceString):
	try:	
		print(sentenceString)
		sentenceString.replace('/', '')
		#sentenceString = sentenceString.decode('utf-8').encode('ascii', 'replace')
		import re
		result=corenlp.parse(sentenceString)
		j=json.loads(result)
		for i in j['sentences']:
			for k in i['words']:
				if k[1]['NamedEntityTag']=='LOCATION':
					return k[0]
		return('NA')
	except:
		return('NA')

df_emails['Location']=df_emails['MetadataSubject'].map(isLocation)


data=pd.read_csv('/home/anantgupta/Documents/Python/MachineLearning/KaggleClinton/withStanfordData.csv')
data=df_emails
locationCount=data['Location'].value_counts()

locationDataTop10=locationCount.head(10)
x_pos = np.arange(len(locationDataTop10.index))
plt.bar(x_pos,locationDataTop10.values)
plt.xlabel('Region')
plt.ylabel('Reference Count')
plt.xticks(x_pos,locationDataTop10.index,rotation=70)
plt.show()



###### End of 2016-05-01

### 2016-05-08 ####
# Similarity between tweets

# We will be using TFID vectorizer
import networkx as nx
import re
import os
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
os.chdir("/home/anantgupta/Documents/Python/MachineLearning/KaggleClinton/")
df_aliases = pd.read_csv('./output/Aliases.csv', index_col=0)
df_emails = pd.read_csv('./output/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('./output/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('./output/Persons.csv', index_col=0)

vect = TfidfVectorizer(min_df=1)
tfidf = vect.fit_transform(df_emails['ExtractedBodyText'])
cosine=(tfidf * tfidf.T).A


#Similarity on a particular country


