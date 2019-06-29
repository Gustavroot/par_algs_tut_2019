from scipy.linalg import lu
import numpy as np
import time
from math import ceil
from scipy.linalg import solve_triangular

# Local dir
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/'

import sys

sys.path.append(dir_path + "../matrix/")

#from matrix import Matrix

"""

IMPORTANT NOTE:

note how the addition of GPU computations decreases the performance
of the code. This is due to so many transfers of single values from
device to host when computing Gaussian-Elimination. The problem is
the way *axpy operations are called, with a value <alpha> that lives
on the host. Try to come up with a way to avoid this, possibilities:

 ** use other Python libraries/functions available, that avoid this
    storage in the <<host>> side
 ** create my own kernels for such *axpy operations with alpha on
    device

"""

#Gateway to all LU decomposition implementations
def lu_decomposer(ijk_form, M, gpu_usage):

    # TODO: improve the logic in the following lines extracting usingMPI
    usingMPI = False
    if type(M) != np.ndarray:
        usingMPI = M.usingMPI
        nb = M.nb
        M = M.mat

    m = M.shape[0]
    n = M.shape[1]

    if m!=n: raise Exception('Implementation of LU only valid here for square matrices')

    N = np.zeros((n,m))

    if ijk_form=="use_scipy":
        start = time.time()
        # ** with scipy first -- lu_mat2 stores L and U in the same matrix
        p, l, u = lu(M)
        end = time.time()
        print("<Scipy-bare LU> elapsed time: " + str(end-start))
        #Store l and u in N
        for i in range(0,n):
            for j in range(i,m):
                N[i,j] = u[i,j]
        for i in range(1,n):
            for j in range(0,i):
                N[i,j] = l[i,j]
        #Return scipy's output as well, for correctness tests
        return (N,p,l,u)

    elif ijk_form=="kij":
        if gpu_usage: print("Turning GPU usage off for this ijk form")
        return kij_lu_decomposer(M)

    elif ijk_form=="kij_opt":
        if gpu_usage:
            return kij_lu_decomposer_opt_gpu(M)
        else:
            return kij_lu_decomposer_opt(M)

    elif ijk_form=="kji":
        if gpu_usage: print("Turning GPU usage off for this ijk form")
        return kji_lu_decomposer(M)

    elif ijk_form=="kji_opt":
        if gpu_usage: print("Turning GPU usage off for this ijk form")
        return kji_lu_decomposer_opt(M)

    elif ijk_form=="ikj_opt":
        if gpu_usage:
            return ikj_lu_decomposer_opt_gpu(M)
        else:
            return ikj_lu_decomposer_opt(M)

    elif ijk_form=="ijk_blocked_opt":
        return ikj_lu_decomposer_blocked_opt(M, nb)

    elif ijk_form=="ijk_opt":
        if gpu_usage:
            return ijk_lu_decomposer_opt_gpu(M)
        else:
            return ijk_lu_decomposer_opt(M)

    elif ijk_form=="jik_opt":
        if gpu_usage: print("Turning GPU usage off for this ijk form")
        return jik_lu_decomposer_opt(M)

    elif ijk_form=="block_right_look_ge":
        if gpu_usage: print("Turning GPU usage off for this ijk form")
        if usingMPI:
            raise Exception("Parallel blocked right-looking Gaussian Elimination currently unavailable.")
        else:
            return block_right_look_ge_sequential(M)

    elif ijk_form=="jki_opt":
        if gpu_usage:
            return jki_lu_decomposer_opt_gpu(M)
        else:
            return jki_lu_decomposer_opt(M)

    else:
        raise Exception('LU decomposer not known')


#LU decomposition, no pivoting, no override of M
def kij_lu_decomposer(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for k in range(0,n):
        for i in range(k+1,n):
            N[i,k] = N[i,k] / N[k,k]
            for j in range(k+1,n):
                N[i,j] = N[i,j] - N[i,k] * N[k,j]

    return N


#LU decomposition, no pivoting, no override of M
def kji_lu_decomposer(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for k in range(0,n):

        for i in range(k+1,n):
            N[i,k] = N[i,k] / N[k,k]

        for j in range(k+1,n):
            for i in range(k+1,n):
                N[i,j] = N[i,j] - N[i,k] * N[k,j]

    return N


#LU decomposition, no pivoting, no override of M
def kji_lu_decomposer_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for k in range(0,n):

        N[k+1:,k] = N[k+1:,k] / N[k,k]

        for j in range(k+1,n):
            N[k+1:,j] = N[k+1:,j] - N[k+1:,k] * N[k,j]

    return N


#LU decomposition, no pivoting, no override of M
def ikj_lu_decomposer_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for i in range(0,n):
        for k in range(0,i):

            N[i,k] = N[i,k] / N[k,k]
            N[i,k+1:] -= N[i,k] * N[k,k+1:]

    return N


#LU decomposition, no pivoting, no override of M
def jki_lu_decomposer_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for j in range(0,n):
        for k in range(0,j):
            N[k+1:,j] -= N[k+1:,k] * N[k,j]
        N[j+1:,j] /= N[j,j]

    return N


#LU decomposition, no pivoting, no override of M
def jik_lu_decomposer_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for j in range(0,n):

        for i in range(0,n):

            N[i,j] -=  N[i,:min(i,j)].dot(N[:min(i,j),j])

        N[j+1:,j] /= N[j,j]

    return N


#LU decomposition, no pivoting, no override of M, implemented on GPU
def jki_lu_decomposer_opt_gpu(M):

    m = M.shape[0]
    n = M.shape[1]

    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray

    from skcuda.cublas import cublasCreate, cublasDaxpy, cublasDscal, cublasDestroy
    import skcuda.misc as misc

    N_gpu = gpuarray.to_gpu(M)

    h = cublasCreate()

    for j in range(0,n):
        for k in range(0,j):

            #N[k+1:,j] = N[k+1:,j] - N[k+1:,k] * N[k,j]
            cublasDaxpy(h, N_gpu[k+1:,k].size, -np.float64(N_gpu[k,j].get()), N_gpu[k+1:,k].gpudata, n, N_gpu[k+1:,j].gpudata, n)

        #N[j+1:,j] /= N[j,j]
        cublasDscal(h, N_gpu[j+1:,j].size, 1.0/np.float64(N_gpu[j,j].get()), N_gpu[j+1:,j].gpudata, n)

    #Move from GPU to CPU
    N = N_gpu.get()

    cublasDestroy(h)

    return N


#LU decomposition, no pivoting, no override of M
def ikj_lu_decomposer_opt_gpu(M):

    m = M.shape[0]
    n = M.shape[1]

    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray

    from skcuda.cublas import cublasCreate, cublasDaxpy, cublasDscal, cublasDestroy
    import skcuda.misc as misc

    N_gpu = gpuarray.to_gpu(M)

    h = cublasCreate()

    for i in range(0,n):
        for k in range(0,i):

            #N[i,k] = N[i,k] / N[k,k]
            cublasDscal(h, N_gpu[i,k].size, 1.0/np.float64(N_gpu[k,k].get()), N_gpu[i,k].gpudata, 1)
            #N[i,k+1:] -= N[i,k] * N[k,k+1:]
            cublasDaxpy(h, N_gpu[k,k+1:].size, -np.float64(N_gpu[i,k].get()), N_gpu[k,k+1:].gpudata, 1, N_gpu[i,k+1:].gpudata, 1)

    #Move from GPU to CPU
    N = N_gpu.get()

    cublasDestroy(h)

    return N


#LU decomposition, no pivoting, no override of M
def kij_lu_decomposer_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    #print_rate = 0.1

    for k in range(0,n):
        #buf_str = "Stage " + str(k) + " of kij alg. " + str(float(k)/float(n)*100) + " %" + ". Resources usage: " + resources_usage()
        #if k%(n*print_rate)==0:
        #    print buf_str
        for i in range(k+1,n):
            N[i,k] = N[i,k] / N[k,k]
            N[i,k+1:] -= N[i,k] * N[k,k+1:]

    return N


#LU decomposition, no pivoting, no override of M
def kij_lu_decomposer_opt_gpu(M):

    m = M.shape[0]
    n = M.shape[1]

    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray

    from skcuda.cublas import cublasCreate, cublasDaxpy, cublasDestroy
    import skcuda.misc as misc

    N_gpu = gpuarray.to_gpu(M)

    h = cublasCreate()

    for k in range(0,n):
        for i in range(k+1,n):
            N_gpu[i,k] = N_gpu[i,k] / N_gpu[k,k]
            #N[i,k+1:] -= N[i,k] * N[k,k+1:]
            cublasDaxpy(h, N_gpu[k,k+1:].size, -np.float64(N_gpu[i,k].get()), N_gpu[k,k+1:].gpudata, 1, N_gpu[i,k+1:].gpudata, 1)

    #Move from GPU to CPU
    N = N_gpu.get()

    cublasDestroy(h)

    return N


#LU decomposition, no pivoting, no override of M
def ijk_lu_decomposer_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    N = np.copy(M)

    for i in range(0,n):
        for j in range(0,n):
            N[i,j] -= N[i,:min(i,j)].dot(N[:min(i,j),j])
            #N_gpu[i,j] -= linalg.dot(N_gpu[i,:min(i,j)], N_gpu[:min(i,j),j])
            if j<i:
                #cublasDscal(h, N_gpu[i,j].size, 1.0/np.float64(N_gpu[j,j].get()), N_gpu[i,j].gpudata, 1)
                N[i,j] /= N[j,j]

    return N


#LU decomposition, no pivoting, no override of M
def ijk_lu_decomposer_opt_gpu(M):

    m = M.shape[0]
    n = M.shape[1]

    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray

    from skcuda.cublas import cublasCreate, cublasDestroy, cublasDdot#, cublasDscal
    import skcuda.misc as misc
    #import skcuda.linalg as linalg
    #linalg.init()

    N_gpu = gpuarray.to_gpu(M)

    h = cublasCreate()

    for i in range(0,n):
        for j in range(0,n):
            #N[i,j] -= N[i,:min(i,j)].dot(N[:min(i,j),j])
            N_gpu[i,j] -= cublasDdot(h, N_gpu[i,:min(i,j)].size, N_gpu[i,:min(i,j)].gpudata, 1, N_gpu[:min(i,j),j].gpudata, n)
            if j<i:
                N_gpu[i,j] /= N_gpu[j,j]
                #cublasDscal(h, N_gpu[i,j].size, 1.0/np.float64(N_gpu[j,j].get()), N_gpu[i,j].gpudata, 1)

    #Move from GPU to CPU
    N = N_gpu.get()

    cublasDestroy(h)

    return N


def ikj_lu_decomposer_blocked_opt(A, nb):

    n = A.shape[1]
    if (n/nb - int(n/nb)) != 0:
        raise Exception("Only supporting nb dividing the matrix dims exactly.")

    N = ceil(n / nb)
    M = np.copy(A)

    # Recursive decomposition
    for i in range(N):

        # Apply an LU decomposition to the ii block matrix
        M[i*nb:(i+1)*nb, i*nb:(i+1)*nb] = jik_lu_decomposer_opt(M[i*nb:(i+1)*nb, i*nb:(i+1)*nb])

        # Apply _TRSM to solve lower part of block column
        mrhs = M[(i+1)*nb:, i*nb:(i+1)*nb].transpose()
        sol = solve_triangular(M[i*nb:(i+1)*nb, i*nb:(i+1)*nb], mrhs, trans=1)
        M[(i+1)*nb:, i*nb:(i+1)*nb] = sol.transpose()

        # Apply _TRSM to solve right part of block row
        M[i*nb:(i+1)*nb, (i+1)*nb:] = solve_triangular(M[i*nb:(i+1)*nb, i*nb:(i+1)*nb], M[i*nb:(i+1)*nb, (i+1)*nb:], lower=True, unit_diagonal=1)

        # Prepare the rest of the block matrix for recursive computation of LU
        M[(i+1)*nb:, (i+1)*nb:] -= np.dot(M[(i+1)*nb:, i*nb:(i+1)*nb], M[i*nb:(i+1)*nb, (i+1)*nb:])

    return M
