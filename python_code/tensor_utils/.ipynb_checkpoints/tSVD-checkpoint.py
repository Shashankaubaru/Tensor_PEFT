import torch
import numpy as np
from tensor_utils import modek_product
import tensorly as tl
tl.set_backend('pytorch')



def _TransposeTensor(A):
    
    n1, n2, n3 = A.size()
    B = torch.zeros((n2, n1, n3), dtype=torch.float)
    B[:,:,0] = A[:,:,0].T
    i=0
    for k in range(n3-1,0,-1):
        i = i+1;
        B[:,:,i] = A[:,:,k].T
             
    return B
        
    
    
def tSVD(A, k = 0):
    """
    This function computes the truncated (or full) tSVD for a given
    tensor A 
    """
    n1, n2, n3 = A.size()
    transpose_flag = 0
        
    if n2 > n1:
        transpose_flag = 1
        A = _TransposeTensor(A)
        tmp =n1
        n1=n2
        n2=tmp
    
    if k ==0:
        full = True
        k = np.minimum(n1,n2)
    else:
        full = False

    A = torch.fft.fft(A, dim = 2)
    U = torch.zeros((n1, k, n3), dtype=torch.cfloat)
    S = torch.zeros((k, k, n3), dtype=torch.cfloat)
    V = torch.zeros((n2, k, n3), dtype=torch.cfloat)
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
        U[:,:,i] = U1
        S[:,:,i] = torch.diag(S1)
        V[:,:,i] = V1
        
    U = torch.fft.ifft(U, dim = 2)
    S = torch.fft.ifft(S, dim = 2)
    V = torch.fft.ifft(V, dim = 2)
    
    if transpose_flag:
        Uold =U
        U= V
        V= Uold
        S = _TransposeTensor(S)
 
    
    return U.type(torch.float), S.type(torch.float), V.type(torch.float)
  
def tSVD_approx(A, k = 0):
    """
    This function computes the truncated (or full) tSVD for a given
    tensor A 
    """
    n1, n2, n3 = A.size()
        
    
    if k ==0:
        full = True
        k = np.minimum(n1,n2)
    else:
        full = False

    A = torch.fft.fft(A, dim = 2)
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

        
    Aout = torch.fft.ifft(Aout, dim = 2)
 
    return Aout.type(torch.float)

def construct_projector(U, k, ny):
    """
    This function computes the projector given left singular tensor U
    """
    
    Z = torch.eye(ny)
    Z = torch.roll(Z,1,0)
    
    U1 = U[:,0,:]
    
    for i in range(1, k):
        U1 = torch.cat((U1, U[:,i,:]),1)
        
    B = U1
    
    for i in range(ny-1):
        B = torch.cat((B,(U1 @ torch.kron(torch.eye(k),Z.matrix_power(i+1)))),0)

    return B
    
    
    