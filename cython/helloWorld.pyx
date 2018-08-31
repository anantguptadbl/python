from __future__ import print_function
cimport numpy as np
ctypedef np.int_t DTYPE_t

def sayHello():
        print("Hello there. This is a cython program")
        
# Execution
#import helloWorld
#helloWorld.sayHello()
