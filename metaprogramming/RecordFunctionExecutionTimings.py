# Recording the timings of each function call
import re
import time

def getFuncTimings(funcName):
    def _method(self,*argl,**argd):
        global indent
        startTime=time.time()
        returnval = getattr(self,'_H_%s' % funcName)(*argl,**argd)
        endTime=time.time()
        print("Elapsed Time for function {} is {} secs".format(funcName,endTime - startTime))
        return returnval
    return _method


class FuncTimings(type):
    def __new__(cls,classname,bases,classdict):
        for funcName,funcPointer in classdict.items():
            if callable(funcPointer):
                classdict['_H_%s'%funcName] = funcPointer    
                classdict[funcName] = getFuncTimings(funcName)

        return type.__new__(cls,classname,bases,classdict)

class Test(object):
    __metaclass__ = FuncTimings

    def __init__(self):
        self.a = 10
    # Dummy
    def meth1(self):pass
    
    # Addition
    def add(self,a,b):return a+b
    
    # Factorial
    def fac(self,val): # Factorial
        if val == 1:
            return 1
        else:
            return val * self.fac(val-1)

if __name__ == '__main__':
    l = Test()
    l.meth1()
    print l.add(1,2)
    print l.fac(10)
