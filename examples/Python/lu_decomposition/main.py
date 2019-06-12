import sys
import numpy as np
import time
#from guppy import hpy

from miscellaneous import read_inp_parms, init_random, resources_usage
from test_bank import basic_test

#------------------------------------------

#Init
inp_params = read_inp_parms()
init_random()

#Unpacking input params
algrthm = inp_params['ge_alg']
n = int(inp_params['n'])
m = int(inp_params['m'])

#------------------------------------------

#APPLY TESTS

start = time.time()
basic_test(algrthm, n, m)
end = time.time()

print

print "Total elapsed time: " + str(end-start)

print
print

print "Some relevant information, on the underlying Numerical Libraries used by Numpy during this run:"

np.__config__.show()

#print resources_usage()
