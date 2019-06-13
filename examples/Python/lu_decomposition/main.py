#First of all, check necessary Python packages
from miscellaneous import check_necess_packs
check_necess_packs()

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
    gpu_use = bool(int(inp_params['GPU_use']))
except KeyError:
    print
    print "Missing one or more params for this run"
    raise Exception(KeyError)

#------------------------------------------

#APPLY TESTS

print
print "Specs of the run:"
print " ** dimensions of matrix = " + str(n) + "x" + str(m)
print " ** algorithm used = " + algrthm

start = time.time()
basic_test(algrthm, n, m)
end = time.time()

print

print "Total elapsed time (including tests): " + str(end-start)

print
print

print "Some relevant information, on the underlying Numerical Libraries used by Numpy (on your machine) during this run:"

np.__config__.show()

#print resources_usage()
