# MetaClass to force docstring to be implemented

class MyMeta(type):
    """
    This is a simple class for declaring a metaclass
    """
    def __new__(meta,name,bases,classDict):
        print("Creating new MetaClass")
        for funcName,funcPointer in classDict.items():
            if(callable(funcPointer)):
                if funcPointer.__doc__ is None:
                    print("The function {} does not have a valid docString. Please update".format(funcName))
                elif len(funcPointer.__doc__) < 10:
                    print("The function {} has a very small doc string. Please update".format(funcName))
        return type.__new__(meta,name,bases,classDict)

def performOpsOnModule():
    for funcName,funcPointer in globals().items():
            if(callable(funcPointer)):
                if funcPointer.__doc__ is None:
                    print("The function {} does not have a valid docString. Please update".format(funcName))
                elif len(funcPointer.__doc__) < 10:
                    print("The function {} has a very small doc string. Please update".format(funcName))
    
class Test1():
    __metaclass__=MyMeta
    
    def __func1__(self):
        a=1
        b=2
        print(a+b)
        
    def __func2__(self):
        """
        """
        a=1
        b=2
        print(a+b)
        
    def __func3__(self):
        """
        We have a sufficiently large docstring
        """
        a=1
        b=2
        print(a+b)
    
def jkl():
    print("sample")
    
    
if __name__=="__main__":
    testObj=Test1()
    performOpsOnModule()
    
