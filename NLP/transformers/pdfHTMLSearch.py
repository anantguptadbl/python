import bs4
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import numpy as np
import pandas as pd

baseFileName="report1.html"

fileName="file://" + baseFileName
browser=webdriver.Chrome("chromedriver")
#path_to_phantomJS='./phantomjs'
#browser = webdriver.PhantomJS(executable_path = path_to_phantomJS,service_log_path=os.path.devnull)
browser.get(fileName)
pageContents=browser.page_source
elementList=[]
for curElement in browser.find_elements_by_xpath("//*"):
    elementList.append([curElement.id,curElement.parent,curElement.tag_name,curElement.location,curElement.size])
    
soup=BeautifulSoup(pageContents,"html.parser")
soup.findAll('style')
import cssutils
selectors = {}
for styles in soup.select('style'):
    css = cssutils.parseString(styles.encode_contents())
    for rule in css:
        if rule.type == rule.STYLE_RULE:
            style = rule.selectorText
            selectors[style] = {}
            for item in rule.style:
                propertyname = item.name
                value = item.value
                selectors[style][propertyname] = value    
allElements=soup.find_all()

browser.close()
browser.quit()

# EXAMPLE 1 : We will ask a simple question and then mark the section
import pandas as pd
soupElements=pd.DataFrame(allElements,columns=['element'])
soupElements['x']=[x[3]['x'] for x in elementList]
soupElements['y']=[x[3]['y'] for x in elementList]
soupElements['h']=[x[4]['height'] for x in elementList]
soupElements['w']=[x[4]['width'] for x in elementList]

def getTextChildren(x):
    childList=list(x.children)
    if(len(childList)==1):
        if(type(childList[0])==bs4.element.NavigableString):
            return(1)
    return(0)

def getPlainText(x):
    childList=list(x.children)
    if(len(childList)==1):
        if(type(childList[0])==bs4.element.NavigableString):
            return(str(childList[0]))
    return('')

soupElements['isPlainText']=soupElements['element'].map(lambda x : getTextChildren(x))
soupElements['plainText']=soupElements['element'].map(lambda x : getPlainText(x))
soupElements=soupElements[soupElements['isPlainText']==1]
soupElements=soupElements.reset_index(drop=True)

# We will find text that are hanging... There is very little text
def hangingText(x):
    if(len(x) < 20):
        return(1)
    else:
        return(0)
soupElements['isHanging']=soupElements['plainText'].map(lambda x : hangingText(x))

def getClosest(curRow):
    x=curRow['x']
    y=curRow['y']
    h=curRow['h']
    w=curRow['w']
    coords=[[x,y],[x,y+h],[x+w,y],[x+w,y+h]]
    minRows=np.min(np.array(list(zip(
    np.linalg.norm(coords[0] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1),
    np.linalg.norm(coords[1] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1),
    np.linalg.norm(coords[2] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1),
    np.linalg.norm(coords[3] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1)
    ))),axis=1)
    closest5=np.argsort(minRows)[0:5]
    closest5EuclideanDistance=np.sort(minRows)[0:5]
    return({'closest5':closest5[1:],'closest5EuclideanDistance':closest5EuclideanDistance[1:]})
    
soupElements['closestIndex']=soupElements.apply(lambda x : getClosest(x),axis=1)
soupElements['closestSingleIndex']=soupElements['closestIndex'].map(lambda x : x['closest5'][0])
soupElements['closestSingleIndexDist']=soupElements['closestIndex'].map(lambda x : x['closest5EuclideanDistance'][0])
soupElements=soupElements.reset_index()

#from sklearn.cluster import KMeans
#%time kmeans = KMeans(n_clusters=50, random_state=42).fit(soupElements[['x','y','h','w']].values)
#soupElements['clusterLabel']=kmeans.labels_

print("Cell Execution Completed")
# CLUBBING TEXT TOGETHER

#soupElements[soupElements['plainText']=='Firmwide']
#print(soupElements[soupElements['plainText']=='Metrics'])

# Merge Data Points logically
group1=1
soupElements['group1']=0

for i,curRow in soupElements.iterrows():
    if(curRow['group1']==0):
        soupElements.loc[(soupElements['index']==curRow['index']) & (soupElements['group1']==0),'group1']=group1
        soupElements.loc[(soupElements['index']==curRow['closestSingleIndex']) & (soupElements['group1']==0),'group1']=group1
        soupElements.loc[(soupElements['closestSingleIndex']==curRow['closestSingleIndex']) & (soupElements['group1']==0),'group1']=group1
        group1=group1+1

# Now we will assign the hanging elements to a non hanging node
def getClosestNonHanging(curRow):
    x=curRow['x']
    y=curRow['y']
    h=curRow['h']
    w=curRow['w']
    coords=[[x,y],[x,y+h],[x+w,y],[x+w,y+h]]
    minRows=np.min(np.array(list(zip(
    np.linalg.norm(coords[0] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1),
    np.linalg.norm(coords[1] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1),
    np.linalg.norm(coords[2] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1),
    np.linalg.norm(coords[3] - np.array(list(zip(soupElements['x'].values,soupElements['y'].values))),axis=1)
    ))),axis=1)
    hangingArray=soupElements['isHanging'].values
    closest=np.argsort(minRows)
    closest=[validIndex for i,validIndex in enumerate(closest) if hangingArray[i]==0]
    return(closest[0])

soupElements['closestNonHangingIndex']=soupElements.apply(lambda x : getClosestNonHanging(x),axis=1)
# Now merge the original groupings with the closest Non Hanging
for i,curRow in soupElements[soupElements['isHanging']==0].iterrows():
    # Find the currentRows group1
    curGroup=curRow['group1']
    # Set this group for all the rows that have this as the closest non hanging
    soupElements.loc[soupElements['closestNonHangingIndex']==curGroup,'group1']=curGroup
    
# Now we will create sentences for BERT
sentences=[]
for curGroup in soupElements['group1'].unique():
    sentences.append(' '.join([x.replace('.','') for x in soupElements[soupElements['group1']==curGroup]['plainText'].values if len(x)<3000]))
    
import torch
question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# The format is paragraph first and then question
corpus = ' . '.join(sentences[0:20])
question = "What is the reported revenue in billions?"
indexed_tokens = question_answering_tokenizer.encode(corpus, question, add_special_tokens=True)
encoded_dict = question_answering_tokenizer(corpus, question)
segment_ids=np.zeros(len(indexed_tokens))
segment_ids[np.where(np.array(indexed_tokens)==102)[0][0]:]=1
segments_tensors = torch.LongTensor([segment_ids])
tokens_tensor = torch.LongTensor([indexed_tokens])

with torch.no_grad():
    start_logits, end_logits = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# get the highest prediction
answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(start_logits):torch.argmax(end_logits)+1])
print(answer)

answerTerms=answer.split(' ')
for i,curRow in soupElements[1:].iterrows():
    countMain=np.sum([1 if curTerm in soupElements.loc[i,'plainText'] else 0 for curTerm in answerTerms])
    countPrior=np.sum([1 if curTerm in soupElements.loc[i-1,'plainText'] else 0 for curTerm in answerTerms])
    countPost=np.sum([1 if curTerm in soupElements.loc[i+1,'plainText'] else 0 for curTerm in answerTerms])
    if(np.sum(countMain + countPrior + countPost) > len(answerTerms)):
        print(i)
        answerI=i
        break
        
resultsCount=[]
for element in [0,1,-1]:
    resultsCount.append(np.sum([1 if curTerm in soupElements.loc[answerI+1+element,'plainText'] else 0 for curTerm in answerTerms]))
    
displayElements=[i for i,x in enumerate(resultsCount) if x > 0]
displayElements=[x+answerI+1 for x in displayElements]

startIndex=pageContents.find('<html')
endIndex=pageContents[startIndex:].find('>')+startIndex
pageContents=pageContents.replace("</html>","")
pageContents=pageContents[endIndex+1:]

for curAnswerElement in soupElements.loc[displayElements]['element'].values:
    print(str(pageContents).find(str(curAnswerElement)))
    pageContents=pageContents.replace(str(curAnswerElement),'<span style="background-color:yellow;">' + str(curAnswerElement) + '</span>')

createNewPage='<html><h1>EMBEDDED PAGE</h1>'+pageContents+'</html'    
with open(baseFileName,"w") as f:
    f.write(createNewPage)
