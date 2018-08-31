import numpy as np
cimport numpy as np

ctypedef np.double_t DTYPE_t

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
                int(*compar)(const_void *, const_void *)) nogil

cdef int mycmp(const_void * pa, const_void * pb):
    cdef double a = (<double *>pa)[0]
    cdef double b = (<double *>pb)[0]
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

cdef void myfunc(double * y, ssize_t l) nogil:
    qsort(y, l, sizeof(double), mycmp)

def sortArray(np.ndarray[DTYPE_t,ndim=1] inpArray):
    cdef int lengthArr=inpArray.shape[0]
    cdef double* ArrPtr=&inpArray[0]
    myfunc(ArrPtr,lengthArr)

# Usage
#import numpy as np
#inpArray=np.array([1.,2.,3.,7.,4.])
#cythonSort.sortArray(inpArray)
#print(inpArray)
# [1. 2. 3. 4. 7.]
