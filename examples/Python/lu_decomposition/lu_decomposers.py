from scipy.linalg import lu
import numpy as np
from miscellaneous import resources_usage

#Gateway to all LU decomposition implementations
def lu_decomposer(ijk_form, M):

    m = M.shape[0]
    n = M.shape[1]

    if m!=n: raise Exception('Implementation of LU only valid here for square matrices')

    N = np.zeros((n,m))

    if ijk_form=="use_scipy":
        p, l, u = lu(M)
        #Store l and u in N
        for i in range(0,n):
            for j in range(i,m):
                N[i,j] = u[i,j]
        for i in range(1,n):
            for j in range(0,i):
                N[i,j] = l[i,j]
        #Return scipy's output as well, for tests
        return (N,p,l,u)

    elif ijk_form=="kij":
        return kij_lu_decomposer(M)

    elif ijk_form=="kji":
        return kji_lu_decomposer(M)

    elif ijk_form=="kji_opt":
        return kji_lu_decomposer_opt(M)

    elif ijk_form=="ikj_opt":
        return ikj_lu_decomposer_opt(M)

    elif ijk_form=="jki_opt":
        return jki_lu_decomposer_opt(M)

    else:
        raise Exception('LU decomposer not known')


#LU decomposition, no pivoting, no override of M
def kij_lu_decomposer(M):

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
