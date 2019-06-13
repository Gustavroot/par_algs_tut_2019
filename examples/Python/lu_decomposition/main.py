#Enforce Python3 usage
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

#First of all, check necessary Python packages
from miscellaneous import check_necess_packs
check_necess_packs() #this doesn't contemplate GPU usage

import sys
import numpy as np
import time

from miscellaneous import read_inp_parms, init_random, resources_usage
from test_bank import basic_test

#------------------------------------------

#Init
inp_params = read_inp_parms()
init_random()

#Unpacking input params
try:
    algrthm = inp_params['ge_alg']
    n = int(inp_params['n'])
    m = int(inp_params['m'])
    gpu_usage = bool(int(inp_params['GPU_use']))
except KeyError:
    print("")
    print("Missing one or more params for this run")
    raise Exception(KeyError)

#Further, if GPU usage is on, check for the necessary packs
#Checking if PyCUDA and Scikit-CUDA installed
packs_to_install = list()
try:
    import pycuda
except ImportError:
    packs_to_install.append("pycuda")
try:
    import skcuda
except ImportError:
    packs_to_install.append("scikit-cuda")
if len(packs_to_install)>0:
    print("The following packages need to be installed:")
    print(packs_to_install)
    raise Exception('SOME PYTHON PACKS WERE NOT FOUND')

#------------------------------------------

#APPLY TESTS

print("")
print("Specs of the run:")
print(" ** dimensions of matrix = " + str(n) + "x" + str(m))
print(" ** algorithm used = " + algrthm)

start = time.time()
basic_test(algrthm, n, m, gpu_usage)
end = time.time()

print("")

print("Total elapsed time (including tests): " + str(end-start))

print("")
print("")

print("Some relevant information, on the underlying Numerical Libraries used by Numpy (on your machine) during this run:")

np.__config__.show()

#print resources_usage()
