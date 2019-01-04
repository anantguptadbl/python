import time
from guppy import hpy
import threading
from datetime import datetime

class memoryGetter(threading.thread):
  def __init(self):
    super(memoryGetter,self).__init__():
    self.memoryData=[]
    self.endFlag=0
    self.h=hpy()
    
  def stop(self):
    self.endFlag=1
    
  def run(self):
    while seld.endFlag==0:
      time.sleep(2)
      self.memoryData.append([datetime.now(),(h.heap().size * 1.00000) / (1024 * 1024)])
      
  def getMemoryProfile(self):
    return self.memoryData
    
curThread=memoryGetter()
curThread.start()
# After some time
curThred.stop()
print(curThread.getMemoryProfile())
