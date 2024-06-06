import torch
import numpy as np
from tensor_utils import tSVD_M, transformation_matrices
import tensorly as tl
from utils import *
tl.set_backend('pytorch')


W = torch.load('tensor_Q.pt')

idx1 = torch.randperm(W.size(0))[:500]
idx2 = torch.randperm(W.size(1))[:500]

W = W[idx1,:,:]
W = W[:,idx2,:]

print(W.size())

nrm = torch.norm(W,p='fro')

M = transformation_matrices.dct_matrix(40)

ranks = range(2,400,10)

err = torch.zeros([len(ranks),1])

j=0
for i in ranks:
    print("rank", i)
    W1 = tSVD_M.tSVD_M_approx(W, M.type(torch.float), k = i)
    err[j] = torch.norm((W - W1), p = 'fro')/nrm
    print(err[j])
    j =j+1

torch.save(err, 'errorsQ.pt')