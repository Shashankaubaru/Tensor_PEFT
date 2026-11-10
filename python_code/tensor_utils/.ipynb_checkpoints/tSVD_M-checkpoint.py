import torch
import numpy as np
from tensor_utils import modek_product
import tensorly as tl
tl.set_backend('pytorch')


def _TransposeTensorM(A,M):
    
    n1, n2, n3 = A.size()
    B = torch.zeros((n2, n1, n3), dtype=torch.float)
    A = modek_product(A, M, k=2)
    B[:,:,0] = A[:,:,0].T
    for k in range(n3):
        B[:,:,k] = A[:,:,k].T
    
    B = modek_product(B, M, k=2, inverse= True)
    return B



def tSVD_M(A,M,k = 0):
    """
    This function computes the truncated (or full) tSVD for a given
    tensor A and transformation matrix M
    """
    n1, n2, n3 = A.size()
    transpose_flag = 0
        
    if n2 > n1:
        transpose_flag = 1
        A = _TransposeTensorM(A,M)
        tmp =n1
        n1=n2
        n2=tmp
    
    if k ==0:
        full = True
        k = np.min(n1,n2)
    else:
        full = False

    A = modek_product(A, M, k=2)
    U = torch.zeros((n1, k, n3), dtype=torch.float)
    S = torch.zeros((k, k, n3), dtype=torch.float)
    V = torch.zeros((n2, k, n3), dtype=torch.float)
    for i in range(n3):
        A1 = A[:,:,i]
        if full == True:
            U1, S1, V1 = torch.svd(A1)
        else:
            U1, S1, V1 = torch.svd(A1)
            U1 = U1[:,:k]
            S1 = S1[:k]
            V1 = V1[:,:k]
 #           U1, S1, V1 = tl.truncated_svd(A1, n_eigenvecs=k)
        U[:,:,i] = U1
        S[:,:,i] = torch.diag(S1)
        V[:,:,i] = V1
        
    U = modek_product(U, M, k=2, inverse= True)
    S = modek_product(S, M, k=2, inverse= True)
    V = modek_product(V, M, k=2, inverse= True)
    
    if transpose_flag:
        Uold =U
        U= V
        V= Uold
        S = _TransposeTensorM(S,M)
    
    return U.type(torch.float), S.type(torch.float), V.type(torch.float)
    
def tSVD_M_approx(A,M,k = 0):
    """
    This function computes the truncated (or full) tSVD for a given
    tensor A and transformation matrix M
    """
    n1, n2, n3 = A.size()
    transpose_flag = 0
        
    if n2 > n1:
        transpose_flag = 1
        A = _TransposeTensorM(A,M)
        tmp =n1
        n1=n2
        n2=tmp
    
    if k ==0:
        full = True
        k = np.min(n1,n2)
    else:
        full = False

    A = modek_product(A, M, k=2)
    Aout = torch.zeros((n1, n2, n3), dtype=torch.cfloat)
    
    for i in range(n3):
        A1 = A[:,:,i]
        if full == True:
            U1, S1, V1 = torch.svd(A1)
        else:
#             U1, S1, V1 = torch.svd(A1)
#             U1 = U1[:,:k]
#             S1 = S1[:k]
#             V1 = V1[:,:k]
            U1, S1, V1 = tl.truncated_svd(A1, n_eigenvecs=k)
        A2 = U1.type(torch.float)@(torch.diag(S1.type(torch.float))@V1.type(torch.float))
        
        Aout[:,:,i] = A2
        
    Aout = modek_product(Aout.type(torch.float), M, k=2, inverse= True)
    
    
    return Aout.type(torch.float)
    
    
def construct_projector_M(U, M):
    """
    This function computes the projector given left singular tensor U
    and transformation matrix M
    """
    n1, k, n3 = U.size()
    hatU = modek_product(U, M, k=2)
    U1 = hatU[:, :, 0]
    
    for i in range(1, n3):
        U1 = torch.block_diag(U1, hatU[:, :, i])

    MtI = torch.kron(M.t().contiguous(), torch.eye(n1))
    MI = torch.kron(M, torch.eye(k))
    return MtI @ (U1 @ MI)


def strideM(r,n):
    """
    Function to create stride permutation matrix
    """
    L = torch.zeros(n,n, dtype=torch.float32)
        
    if np.mod(n,r) != 0:
        print("r must divide n")
        return
    else:
        L[n-1,n-1] = 1
        j = np.arange(n-1)
        i = np.mod(j*r, n-1)
        for k in range(n-1):
            L[k,i[k]] = 1
    
    return L
    
      
    


    