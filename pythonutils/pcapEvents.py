import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as bs
import pickle
import urllib3
import json
import glob
import os
import subprocess

process = subprocess.Popen("cmd", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE);

# Main driver
driver=webdriver.Chrome("C:\\Users\\eapganu\\Desktop\\Office\\AIMProject\\gaia-aimincidentresolver\\Data\\chromedriver.exe")
driver.get("https://www.youtube.com/")

# Event data
eventData=[]

# Wait Element
wait=WebDriverWait(driver,15)

# Trigger the first video
driver.find_element_by_xpath("//a[@id='thumbnail']").click()

# There are two possibilities
# 1) There is no advertisement
# 2) There is advertisement

# Check whether there is advertisement element
time.sleep(5)
currentURL=driver.current_url
advPresent=0
try:
    driver.find_element_by_xpath("//span[@class='ytp-ad-preview-container countdown-next-to-thumbnail']")
    advPresent=1
except:
    advPresent=0
if(advPresent==1):
    # There is advertisement
    print("There is advertisement")
    wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@class='ytp-ad-skip-button ytp-button']")))
    # Skip Advertisement
    print("The wait is over. We will now be clicking")
    driver.find_element_by_xpath("//button[@class='ytp-ad-skip-button ytp-button']").click()
else:
    # Nothing to do. The video has started
    print("There is no advertisement")

print("We have come out of the if")
