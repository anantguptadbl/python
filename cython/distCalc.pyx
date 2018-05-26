# distCalc.pyx

# Imports
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# This is just a C syntax to create aliases
DTYPE=np.int
ctypedef np.int_t DTYPE_t

# Creating a function pointer
ctypedef double (*metric_ptr)(double*,double*,int)

# Directly reading it as a native C array
def distCalc(double[:,::1] inpArray not None,DTYPE_t numRows, DTYPE_t numCols):
	# Declaring the resultArray also as a native C array
	cdef double[:] resultArray = np.empty( int(numRows * (numRows-1) * 0.5), dtype='double')
	# Initalizing the counters with the corresponding data types
	cdef int rowIndex,colIndex,rowForwardIndex
	cdef int resultArrayCounter=0
	
	# Function Pointer
	cdef metric_ptr dist_func
	dist_func = &getDist

	# Marking the starting address of the resultArray
	cdef double* Dptr=&resultArray[0]
	# Marking the starting address of the inputArray. This will be used for incrementing and reading the data
	cdef double* Xptr=&inpArray[0,0]

	# Pointer Values of the two segments
	for rowIndex in range(numRows-1):
		for rowForwardIndex in range(rowIndex+1,numRows):
			# Instead of slicing the array, the pointer address increment is sent to the function
			Dptr[resultArrayCounter]=dist_func(Xptr + rowIndex*numCols,Xptr + rowForwardIndex*numCols,numCols)
			resultArrayCounter+=1
	return resultArray

# Reading pointer location ( memory buffer ) instead of accepting an array ( defined structure)
cdef double getDist(double* arr1,double* arr2,int numCols):
	cdef double distVal=0.0;
	cdef int colIndex
	cdef double tmp

	for colIndex in range(numCols):
		tmp=arr1[colIndex]-arr2[colIndex]
		distVal+= tmp*tmp
	return sqrt(distVal)

# Example Run

# The UNIX command that we need to run
# python setup.py build_ext --inplace

#import numpy as np
#import distCalc
#data=np.random.rand(10000,5)
#%time distCalc.distCalc(data,10000,5)

#CPU times: user 324 ms, sys: 276 ms, total: 600 ms
#Wall time: 602 ms
