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

