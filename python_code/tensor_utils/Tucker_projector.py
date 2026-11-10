import torch
import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
tl.set_backend('pytorch')
   
    
def Tucker_projector(A, rank = [1,1,1]):
    """
    This function computes the truncated Tucker for a given
    tensor A 
    """
    n1, n2, n3 = A.size()
    
    core, factors = tucker(A, rank= rank)
    
    Q = torch.kron(factors[2], factors[0])
    
    return Q
    
