import numpy as np
import time

from matrix_factory import random_mat
from lu_decomposers import lu_decomposer
from mat_operations import l_times_u_opt, l_times_u

def basic_test(algrthm, n, m, gpu_usage):

    #Creation of the matrix to be used for tests
    test_rand_mat = random_mat(n,m)

    #Solving both cases: with scipy and our own implementation

    print("")

    start = time.time()
    # ** with scipy first -- lu_mat2 stores L and U in the same matrix
    lu_mat2, p_scp, l_scp, u_scp = lu_decomposer("use_scipy", test_rand_mat, gpu_usage)
    end = time.time()
    print("<Scipy-wrapped LU> elapsed time: " + str(end-start))

    start = time.time()
    # ** then, perform LU decomposition with the chosen implementation
    lu_mat3 = lu_decomposer(algrthm, test_rand_mat, gpu_usage)
    end = time.time()
    print("<Our own " + algrthm + " LU> elapsed time: " + str(end-start))

    print("")

    start = time.time()
    #Check for correctness of the construction of lu_mat2 from scipy's L and U
    assrtn = True
    for i in range(0,n):
        for j in range(i,m):
            assrtn = (assrtn and (lu_mat2[i,j]==u_scp[i,j]))
    for i in range(1,n):
        for j in range(0,i):
            assrtn = (assrtn and (lu_mat2[i,j]==l_scp[i,j]))
    assert assrtn, "Movement of LU from Scipy to lu_mat2 not done correctly"
    end = time.time()
    print("<Test 1 LU> elapsed time: " + str(end-start))

    start = time.time()
    #Check that L*U, as returned by Scipy, gives back the original matrix
    lu_prod_from_scipy = l_times_u_opt(lu_mat2)
    full_LU = np.dot(p_scp, lu_prod_from_scipy) #A = P * L * U
    assert np.allclose(test_rand_mat - full_LU, np.zeros((n,m))), \
           "LU decomposition from Scipy not analyzed/extracted correctly"
    end = time.time()
    print("<Test 2 LU> elapsed time: " + str(end-start))

    start = time.time()
    #Check for correctness of our own LU decomposition, i.e. do we get M back from L * U ?
    lu_prod_from_ownimpl = l_times_u_opt(lu_mat3)
    ERR_MSG = "The LU decomposer " + algrthm + " is not working correctly"
    assert np.allclose(test_rand_mat - lu_prod_from_ownimpl, np.zeros((n,m))), \
           ERR_MSG
    end = time.time()
    print("<Test 3 LU> elapsed time: " + str(end-start))
