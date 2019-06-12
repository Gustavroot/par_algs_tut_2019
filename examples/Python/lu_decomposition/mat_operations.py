import numpy as np

#This function computes L*U, both L and U stored within M, and returns the resulting product
def l_times_u(M):

    m = M.shape[0]
    n = M.shape[1]

    #TODO: generalize for p
    p = n

    N = np.zeros((n,m))

    for i in range(0,n):
        for j in range(0,m):
            min_indx = min(i,j)

            for k in range(0,min_indx):
                N[i,j] += M[i,k] * M[k,j]

            #Due to both L and U being stored within M, careful with the 1's of L
            if i>j:
                N[i,j] += M[i,j] * M[j,j]
            elif j>i:
                N[i,j] += 1.0 * M[i,j]
            else:
                N[i,j] += 1.0 * M[j,j]

    return N


#This function computes L*U, both L and U stored within M, and returns the resulting product
def l_times_u_opt(M):

    m = M.shape[0]
    n = M.shape[1]

    #TODO: generalize for p
    p = n

    N = np.zeros((n,m))

    for i in range(0,n):
        for j in range(0,m):
            min_indx = min(i,j)

            N[i,j] += np.dot(M[i,:min_indx], M[:min_indx,j])

            #Due to both L and U being stored within M, careful with the 1's of L
            if i>j:
                N[i,j] += M[i,j] * M[j,j]
            elif j>i:
                N[i,j] += 1.0 * M[i,j]
            else:
                N[i,j] += 1.0 * M[j,j]

    return N
