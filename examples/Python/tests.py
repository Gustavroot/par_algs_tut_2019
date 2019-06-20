# Enforce Python3 usage
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# Local dir
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/'

import sys
import numpy as np

# Add solver resources to the Python PATH
sys.path.append(dir_path + 'solver/')
from solver import Solver
# Add matrix resources to the Python PATH
sys.path.append(dir_path + 'matrix/')
from matrix import Matrix
# Add resources from previous LU Decomp implementation
sys.path.append(dir_path + 'lu_decomposition/')
from miscellaneous import init_random, check_necess_packs, read_inp_parms

#----------------

# Initializations and extraction of input params
init_random()
check_necess_packs()
inp_params = read_inp_parms('lu_decomposition/input_params.txt')

# Building the Matrix to use in tests
params_matrix = dict()
params_matrix['nx'] = 256
params_matrix['ny'] = 256
A = Matrix(params_matrix, 0)

# Perform LU decomposition through multiple methods and check if correct
# TODO: move these tests to perform Unit Tests
#algos = ['kij', 'kij_opt', 'kji', 'kji_opt', 'ikj_opt', 'ijk_opt', 'jik_opt', 'jki_opt', 'block_right_look_ge']
algos = ['kij', 'kij_opt', 'kji', 'kji_opt', 'ikj_opt', 'ijk_opt', 'jik_opt', 'jki_opt']
for alg in algos:
    print("----")
    #LUdecomp_algorithm = inp_params['ge_alg']
    # Do an appropriate cast for 'GPU_use': first into int, then bool
    gpu_usage = bool(int(inp_params['GPU_use']))
    exec_time = A.computeLU(alg, gpu_usage)
    print("Exec time for LU using " + alg + ": " + str(exec_time))
    A.testLU()

# Call the solver for Ax=b
b = np.ones(A.ny)
params_solver = dict()
solver = Solver(params_solver)
solver.solve(A, b, b)
