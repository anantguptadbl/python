import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.keys import Keys
import pickle
import urllib3
import json
import glob
import os
import subprocess
import signal
import time
from selenium.webdriver.common.action_chains import ActionChains
import psutil

import subprocess
def getEvents(driver,eventData,wait):
        # There are two possibilities
        # 1) There is no advertisement
        # 2) There is advertisement
        eventData.append([time.time(),"VIDEO START",""])
        # Check whether there is advertisement element
        time.sleep(5)
        currentURL=driver.current_url
        eventData.append([time.time(),"URL NAME",currentURL])
        advertisementPresent=0
        while(advertisementPresent !=2):
            try:
                driver.find_element_by_xpath("//span[@class='ytp-ad-preview-container countdown-next-to-thumbnail']")
                #print("There is advertisement")
                eventData.append([time.time(),"ADVERTISEMENT CONTINUING",currentURL])
                wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@class='ytp-ad-skip-button ytp-button']")))
                # Skip Advertisement
                #print("The wait is over. We will now be clicking")
                driver.find_element_by_xpath("//button[@class='ytp-ad-skip-button ytp-button']").click()
                eventData.append([time.time(),"ADVERTISEMENT ENDED",currentURL])
                advertisementPresent=1
            except:
                print("There is no advertisement")
                advertisementPresent=2

        #print("Capturing multiple events")
        #print("Activating the progress bar")
        # We will move to the progress bar
        #driver.find_element_by_xpath("//div[@class='ytp-scrubber-container']").click()
        #driver.find_element_by_xpath("//div[@class='html5-video-container']").click()
        # EVENT 1 : PAUSE AND THEN PLAY AFTER n SECONDS
        def pausePlay(driver,secondsWait):
            driver.find_element_by_xpath("//div[@id='movie_player']").click()
            time.sleep(secondsWait)
            driver.find_element_by_xpath("//div[@id='movie_player']").click()

        #for secondsWait in [1,2,3,4,5]:
        #    eventData.append([time.time(),"PAUSE FOR {0} seconds".format(secondsWait),currentURL])
        #    print("Pause for {0} seconds".format(secondsWait))
        #    pausePlay(driver,secondsWait)
        #    eventData.append([time.time(),"PLAY AFTER PAUSE FOR {0} seconds".format(secondsWait),currentURL])
        #    print("Pause completed for {0} seconds".format(secondsWait))
        #    time.sleep(3)

        # EVENT 2 : FORWARD SEEK for 5*n seconds
        def seekForward(driver,secondsWait5Multiples):
            for i in range(secondsWait5Multiples):
                driver.find_element_by_xpath("//div[@id='movie_player']").send_keys(Keys.ARROW_RIGHT)
        time.sleep(3)
        for secondsWait5Multiples in [1,2,3,4,5]:
            #print("Seek for {0} seconds".format(secondsWait5Multiples))
            eventData.append([time.time(),"SEEKING FORWARD for 5*{0} seconds".format(secondsWait5Multiples),currentURL])
            seekForward(driver,secondsWait5Multiples)
            eventData.append([time.time(),"SEEKED FORWARD for 5*{0} seconds".format(secondsWait5Multiples),currentURL])
            #print("Seek completed for {0} seconds".format(secondsWait5Multiples))
            time.sleep(2)
        eventData.append([time.time(),"CONTINUING 5 SECONDS"])
        time.sleep(5)
        eventData.append([time.time(),"CONTINUED COMPLETED 5 SECONDS"])
        return(currentURL)

def fullDriverFunction(eventData,currentURL):
    print("Starting the chrome driver")
    # Web Driver
    driver=webdriver.Chrome("chromedriver")
    actions = ActionChains(driver)
    wait=WebDriverWait(driver,15)
    
    # Trigger the video
    if(currentURL==''):
        driver.get("https://www.youtube.com/")
        driver.find_element_by_xpath("//a[@id='thumbnail']").click()
    else:
        driver.get(currentURL)
        eventData.append([time.time(),"IGNORE START",""])
        time.sleep(5)
        driver.find_element_by_xpath("//a[@id='thumbnail']").click()
        eventData.append([time.time(),"IGNORE END",""])
    
    # Get the events
    #eventData.append([time.time(),"Opened Youtube Website",""])
    #getEvents(driver,eventData,wait)

    # NExt video within same session
    numVideos=2
    for curVideo in range(numVideos):
        # We will check whether there are some advertisements
        #try:
        #    driver.find_element_by_xpath("//span[@class='ytp-ad-preview-container countdown-next-to-thumbnail']")
        #    print("There is advertisement")
        #    eventData.append([time.time(),"ADVERTISEMENT CONTINUING",currentURL])
        #    wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@class='ytp-ad-skip-button ytp-button']")))
        #    # Skip Advertisement
        #    print("The wait is over. We will now be clicking")
        #    driver.find_element_by_xpath("//button[@class='ytp-ad-skip-button ytp-button']").click()
        #    eventData.append([time.time(),"ADVERTISEMENT ENDED",currentURL])
        #    advertisementPresent=1
        #driver.find_element_by_xpath("//span[@class='view-count style-scope yt-view-count-renderer']").click()
        #driver.find_element_by_xpath("//a[@class='ytp-next-button ytp-button']").click()
        actions.send_keys(Keys.SHIFT,"n")
        actions.perform()
        #driver.find_element_by_xpath("//div[@id='movie_player']").send_keys(Keys.SHIFT,"n")
        eventData.append([time.time(),"VIDEO START",""])
        eventData.append([time.time(),"SLEEP 2 START",""])
        time.sleep(2)
        eventData.append([time.time(),"SLEEP 2 END",""])
        try:
            driver.find_element_by_xpath("//paper-button[@id='button']").click()
            #print("There is some holdup")
            eventData.append([time.time(),"YOUTUBE PROMPT RESOLVED",''])
        except:
            a=1
            #print("There is no holdup")
        currentURL=getEvents(driver,eventData,wait)
        time.sleep(3)
    driver.close()
    return(currentURL)
 
# File Index
currentURL="https://www.youtube.com/watch?v=caEfgyFv4SM&list=PLMRKdK25AuPVjHl9Kdb-gkBy0Cm7Zi2xo"
currentURL="https://www.youtube.com/watch?v=fPO76Jlnz6c&list=PL7DA3D097D6FDBC02"
for fileIndex in range(7):
    try:
        eventData=[]
        process = subprocess.Popen(["tcpdump -ttt -w youtubeFileBollyOngoingPrimeVideo_{0}.dump \n".format(fileIndex)], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        curPID=process.pid
        print("The current PID is {0}".format(curPID))
        for x in range(10):
            currentURL=fullDriverFunction(eventData,currentURL)
    except Exception as e:
        print("We will have to restart with error {0}".format(e))
    finally:
        parent = psutil.Process(curPID)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()
        parent.kill()
        #p = psutil.Process(curPID)
        #p.terminate()  #or p.kill()
        #process.send_signal(signal.SIGINT)
        #process.stdin.flush()
        #process.kill()
        #result = subprocess.check_output("ps -aef | grep tcpdump | grep ttt | cut -d' ' -f3", shell=True)
        #print(result)
        #try:
        #    result=int(str(result).split("\\n")[0].replace("b'",''))
        #    print("The result is {0}".format(result))
        #    os.system("kill -9 {0}".format(result))
        #except:
        #    print("Process kill error. Ignoring")
        print("The PID {0} has been killed".format(curPID))
        #os.kill(curPID, signal.SIGTERM) #or signal.SIGKILL
        import pickle
        pickle.dump(eventData,open("youtubeEventBollyOngoingPrimeVideo_{0}.pickle".format(fileIndex),"wb"))
print("We have completed. We will now analyse the pcap file")
