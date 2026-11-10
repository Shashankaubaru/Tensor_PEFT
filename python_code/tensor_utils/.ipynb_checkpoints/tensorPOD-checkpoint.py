import torch
import numpy as np
from tensor_utils.tSVD import tSVD, construct_projector
import tensorly as tl
tl.set_backend('pytorch')


def _matrix_to_tensor(Y, nx, ny, ns):
    """
    From a matrix Y (nx-x-ny, ns) to a tensor tY (nx, ns, ny)
    """
    tY = torch.zeros((nx, ns, ny), dtype=torch.float32)
    
    for i in range(ns):
        tY[:, i, :] = torch.reshape(Y[:, i], (ny, nx)).T
    return tY

def _tensor_to_matrix(tY, nx, ny, ns):
    """
    From a tensor tY (nx, ns, ny) to a matrix Y (nx-x-ny, ns)
    """
    return tY.permute(2, 0, 1).reshape(nx * ny, ns)


def tensorPOD_oneside(Y, k, nx, ny, ns, project=True):
    """
    This function computes the one-sided tensor POD projection of
    snapshot matrix Y.
    Y can be either a 2D tensor (nx-x-ny, ns) or a 3D tensor (nx, ns, ny).
    """
    if Y.ndim == 2:
        tY = _matrix_to_tensor(Y, nx, ny, ns)
    elif Y.ndim == 3:
        tY = Y
        Y = _tensor_to_matrix(Y, nx, ny, ns)

    # this is not orthogonal after vectorization, the slices are orthogonal though.
    tW = tSVD(tY,k)[0]

    # this is eq (8) FM(U) of the draft. also check line 226. Here the columns are orthogonal.
    # essentially the slices are put in a bigger block matrix
    W_c = construct_projector(tW, k, ny)
    if project:
        return W_c.T @ Y, W_c
    return W_c


def tensorPOD_twoside(Y, k1, k2, nx, ny, ns, project=True):
    """
    This function computes the two-sided tensor POD projection of
    snapshot matrix Y and transformation matrix M.
    Y can be either a 2D tensor (nx-x-ny, ns) or a 3D tensor (nx, ns, ny).
    """
    C, W_c = tensorPOD_oneside(Y, k1, nx, ny, ns, project=True)
    tC = _matrix_to_tensor(C, nx, k1, ns)
            
    tU = tSVD(tC, k2)[0]
    U_c = construct_projector(tU, k2,k1)
        
    Q = W_c @ U_c
    if project:
        return U_c.T @ C, Q
    return  Q        
