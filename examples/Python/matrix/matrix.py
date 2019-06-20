# Local dir
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/'

import sys
# Add resources from previous LU Decomp implementation
sys.path.append(dir_path + '../lu_decomposition/')

from lu_decomposers import lu_decomposer
from mat_operations import l_times_u_opt

import numpy as np
import time

class Matrix:

    def __init__(self, params, A):

        # Default params
        self.isSparse = False
        self.isHermitian = False
        self.isSymmetric = False
        self.isReal = True
        self.isSquare = False
        self.usingMPI = False
        self.wasCholeskyApplied = False
        self.wasQRApplied = False
        self.wasLUApplied = False
        # The param self.blockedStorage makes sense in case of usingMPI being True
        self.blockedStorage = False
        self.verbosity = "FULL"
        self.measureTimings = True

        if 'nx' not in params:
            raise Exception("Dimension nx of the matrix is needed.")
        else:
            self.nx = params['nx']
        if 'ny' not in params:
            raise Exception("Dimension ny of the matrix is needed.")
        else:
            self.ny = params['ny']

        # Unpacking more params
        if 'usingMPI' in params:
            self.usingMPI = params['usingMPI']

        # TODO: add more checks here for self.nx and self.ny

        self.isSquare = (self.nx == self.ny)

        # The possible values of params['buildMatrixFrom'] are: {'random', 'given', }
        if ('buildMatrixFrom' not in params) or (params['buildMatrixFrom']=='given'):
            # In case no specification as to where to take self.mat from, use the given A
            if type(A) is np.ndarray:
                # In case A is a numpy array, but the dimensions are not the same as given, slice
                if A.shape(0)!=self.nx or A.shape(1)!=self.ny:
                    # Slice
                    if self.verbosity == "FULL":
                        print("\nThe given matrix is ok, but we're slicing it - according to your given nx and ny.")
                    self.mat = A[:self.nx,:self.ny]
                else:
                    # Copy the given matrix A
                    self.mat = A[:,:]
            else:
                if self.verbosity == "FULL":
                    print("\nThe param 'buildMatrixFrom' was not specified, and the given matrix is not a NumPy array. Constructing a random matrix for you.")
                self.mat = np.random.rand(self.nx, self.ny)
        elif params['buildMatrixFrom'] == 'random':
            self.mat = np.random.rand(self.nx, self.ny)
        else:
            raise Exception("Give value for matrix param 'buildMatrixFrom' not valid.")




    # Returns the execution time, with a value of zero if self.measureTimings was set to False
    def computeLU(self, algrthm, gpu_usage):

        exec_time = 0

        if self.measureTimings:
            exec_time = time.time()

        self.matLU = lu_decomposer(algrthm, self, gpu_usage)
        self.wasLUApplied = True

        if self.measureTimings:
            exec_time = time.time() - exec_time

        return exec_time


    def getLU(self):
        return self.matLU


    def testLU(self):
        if not self.wasLUApplied:
            print("LU hasn't been applied on this matrix, cannot apply this test.")
        else:
            l_times_u = l_times_u_opt(self.matLU)
            assrt = np.allclose(self.mat - l_times_u, np.zeros((self.nx,self.ny)))
            if assrt:
                print("\nThe LU decomposition was performed correctly.")
            else:
                print("The LU decomposition is not correct, i.e. L*U != A")


    def computeChokesly(self):
        raise Exception("Method computeCholesky() of matrix under construction.")


    def getCholesky(self):
        raise Exception("Method getCholesky() of matrix under construction.")


    def testCholesky(self):
        raise Exception("Method testCholesky() of matrix under construction.")


    def computeQR(self):
        raise Exception("Method computeQR() of matrix under construction.")


    def getQR(self):
        raise Exception("Method getQR() of matrix under construction.")


    def testQR(self):
        raise Exception("Method testQR() of matrix under construction.")


    def display(self):
        print(self.mat)
