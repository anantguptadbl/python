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
ctypedef void (*insertionsort_ptr)(double*,int,double)
ctypedef void (*insertsorted_ptr)(double*,int,double)

# Directly reading it as a native C array
def knn(double[:,::1] inpArray not None,DTYPE_t numRows, DTYPE_t numCols, DTYPE_t numNeighbours):

	# Initalizing the counters with the corresponding data types
	cdef int rowIndex,colIndex,rowForwardIndex
	cdef int resultArrayCounter=0
	
	# Function Pointer
	cdef metric_ptr dist_func
	dist_func = &getDist
	cdef insertionsort_ptr knn_func
	knn_func=&insertIntoPosition
	cdef insertsorted_ptr knn_func1
	knn_func1=&insertIntoPositionFixedArray

	# Marking the starting address of the resultArray
	cdef double[:] resultArray = np.empty(int(numRows) * int(numNeighbours), dtype='double')
	cdef double* Dptr=&resultArray[0]
	cdef int currMaxCounter=0
	cdef double curDist

	# Marking the starting address of the inputArray. This will be used for incrementing and reading the data
	cdef double* Xptr=&inpArray[0,0]

	# Pointer Values of the two segments
	for rowIndex in range(numRows-1):
		currMaxCounter=0
		for rowForwardIndex in range(numRows-1):
			if(currMaxCounter >= 0 and currMaxCounter < numNeighbours):
				curDist=dist_func(Xptr + rowIndex*numCols,Xptr + rowForwardIndex*numCols,numCols)
				knn_func(Dptr + rowIndex*numNeighbours,currMaxCounter,curDist)
				currMaxCounter=currMaxCounter + 1
			elif curDist < resultArray[currMaxCounter]:
				knn_func1(Dptr + rowIndex*numNeighbours,currMaxCounter,curDist)
	return np.asarray(resultArray)

cdef void insertIntoPosition123(double* arr1,int maxCounter,double insertValue):
	cdef int curPointer=1	
	cdef int prevPointer=0
	cdef double curVal
	while curPointer < maxCounter:
		prevPointer=curPointer-1
		curVal=arr1[curPointer]
		while prevPointer >=0 and arr1[prevPointer] > curVal:
			arr1[prevPointer+1]=arr1[prevPointer]
			prevPointer=prevPointer-1					
		arr1[prevPointer+1]=curVal
		curPointer=curPointer+1

cdef void insertIntoPosition(double* arr1,int maxCounter,double insertValue):
	cdef int prevPointer=0
	cdef double curVal
	if insertValue >= arr1[maxCounter]:
		arr1[maxCounter+1]=insertValue
		return
	while prevPointer <= maxCounter:
		if insertValue > arr1[prevPointer] and prevPointer <= maxCounter and insertValue  <= arr1[prevPointer +1]:
			switchCounter=maxCounter + 1
			while switchCounter >= prevPointer + 2:
				arr1[switchCounter]=arr1[switchCounter -1]
				switchCounter=switchCounter-1
			arr1[prevPointer+1]=insertValue
			break
		prevPointer = prevPointer + 1

cdef void insertIntoPositionFixedArray(double* arr1,int maxCounter,double insertValue):
	cdef int prevPointer=maxCounter-1
	cdef double curVal
	while prevPointer <= maxCounter:
		if insertValue > arr1[prevPointer] and prevPointer <= maxCounter and insertValue  <= arr1[prevPointer +1]:
			switchCounter=maxCounter
			while switchCounter >= prevPointer + 1:
				arr1[switchCounter]=arr1[switchCounter -1]
				switchCounter=switchCounter-1
			arr1[prevPointer+1]=insertValue
			break
		prevPointer = prevPointer + 1

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
