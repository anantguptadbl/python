# Writing CYTHON Programs
We need to do the following steps to run cython programs

## Create the someName.pyx file which contains the actual code

## Update the setup.py file
```
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
ext_modules=cythonize("helloWorld.pyx",annotate=True),include_dirs=[numpy.get_include()]
)
```

## Build the .pyx file
We need to build the someName.pyx file using the following command
python setup.py build_ext --inplace

## Use the Module
import someName
