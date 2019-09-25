import traceback
import requests
import json
from bs4 import BeautifulSoup 

def getErrorResolution(e):
	errorTraceback=traceback_str = ''.join(traceback.format_tb(e.__traceback__))
	errorString=str(e)
	errorStringHTML=errorString.replace(" ","%20")
	issueLink="https://stackoverflow.com/questions/20441035/unsupported-operand-types-for-int-and-str"
	#results=requests.get("https://api.stackexchange.com/2.2/search?order=desc&sort=activity&intitle={0}&site=stackoverflow".format(errorStringHTML))
	#jsonData=json.loads(results.text)
	#issueLink=jsonData['items'][0]['link']
	#curQuestion=jsonData['items'][0]['question_id']
	#questionResults=requests.get("https://api.stackexchange.com/2.2/questions/{0}/answers?order=desc&sort=activity&site=stackoverflow&filter=withbody".format(curQuestion))
	#questionJsonData=json.loads(questionResults.text)
	#contentBody=questionJsonData['items'][0]['body']
	#soup = BeautifulSoup(contentBody, 'html') 
	htmlString="<h2> ERROR </h2> </br> <b color='red'> {0} </b> <h2> SOLUTION </h2> </br> Explicit int to str conversion </br> <h2> Issue Link </h2> </br> <a href='{1}'> StackOverflow Link</a>".format(errorString,issueLink)
	return(htmlString)