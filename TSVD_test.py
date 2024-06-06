import torch
import numpy as np
from tensor_utils import tSVD
import tensorly as tl
from utils import *
tl.set_backend('pytorch')


W = torch.load('tensor_Q.pt')

# idx1 = torch.randperm(W.size(0))[:500]
# idx2 = torch.randperm(W.size(1))[:500]

# W = W[idx1,:,:]
# W = W[:,idx2,:]

print(W.size())

nrm = torch.norm(W,p='fro')

ranks = range(2,40,2)

err = torch.zeros([len(ranks),1])

j=0
for i in ranks:
    print("rank", i)
    W1 = tSVD.tSVD_approx(W, k = i)
    err[j] = torch.norm((W - W1), p = 'fro')/nrm
    print(err[j])
    j =j+1

torch.save(err, 'errorsQ.pt')