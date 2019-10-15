# File : distCalc2.pyx

# Imports
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

# This is just a C syntax to create aliases
DTYPE=np.int
ctypedef np.int_t DTYPE_t

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    np.ulong_t index
    np.float64_t value

cdef int _compare(const_void *a, const_void *b):
    cdef np.float64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1

cdef argsort(np.float64_t* data, np.int_t orderLimit, np.int_t numRows,int* order):
    cdef np.ulong_t i
    cdef np.ulong_t n = numRows
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(orderLimit):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)

# Creating a function pointer
ctypedef void (*metric_ptr)(double*,int,double*,double*,int)

# Reading pointer location ( memory buffer ) instead of accepting an array ( defined structure)
cdef void getDist(double* val,int rowForwardIndex,double* arr1,double* arr2,int numCols):
    cdef double distVal=0.0;
    cdef int colIndex
    cdef double tmp
    for colIndex in range(numCols):
        tmp=arr1[colIndex]-arr2[colIndex]
        distVal+= tmp*tmp
    val[rowForwardIndex]=distVal
    #return 1
    #return sqrt(distVal)

# Directly reading it as a native C array
def distCalc(double[:,::1] inpArray not None,DTYPE_t numRows, DTYPE_t numCols):
    # We will create the temp array which will store the rows 
    cdef np.float64_t[:] tempArray = np.empty(numRows,dtype='double')
    cdef int[:] tempArrayOrderResult = np.empty((numRows*50),dtype=np.int)
    # Initalizing the counters with the corresponding data types
    cdef int rowIndex,colIndex,rowForwardIndex

    # Function Pointer
    cdef metric_ptr dist_func
    dist_func = &getDist

    # Marking the starting address of the inputArray. This will be used for incrementing and reading the data
    cdef double* Xptr=&inpArray[0,0]
    # Marking the starting address of the tempOutputArray which we will input into the argsort
    cdef double* Tptr=&tempArray[0]
    cdef int* Rptr=&tempArrayOrderResult[0]

    # Pointer Values of the two segments
    # We will also use the argSort functionality here
    
    for rowIndex in range(numRows):
        for rowForwardIndex in range(numRows):
            # Instead of slicing the array, the pointer address increment is sent to the function
            #Tptr[rowForwardIndex]=dist_func(Xptr + rowIndex*numCols,Xptr + rowForwardIndex*numCols,numCols)
            dist_func(Tptr,rowForwardIndex,Xptr + rowIndex*numCols,Xptr + rowForwardIndex*numCols,numCols)
        # Now we will do the argsort
        argsort(Tptr,50,numRows,Rptr + 50*rowIndex)
    return tempArrayOrderResult
    
# Example Run    
#import numpy as np
#import distCalc2
#import scipy.spatial.distance

#numRows=100000
#data=np.random.rand(numRows,300).astype(np.float32)
#%time results=distCalc2.distCalc(data[0:numRows].astype(np.double),numRows,300)
#results=np.asarray(results)
#results=results.reshape(-1,50)
#print(results.shape)
