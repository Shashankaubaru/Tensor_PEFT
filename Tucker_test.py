import torch
import numpy as np
from tensorly.decomposition import tucker
import tensortools as tt
import tensorly as tl
from utils import *
tl.set_backend('pytorch')


W = torch.load('tensor.pt')

idx1 = torch.randperm(W.size(0))[:500]
idx2 = torch.randperm(W.size(1))[:500]

W = W[idx1,:,:]
W = W[:,idx2,:]

print(W.size())

nrm = torch.norm(W,p='fro')

ranks = range(2,21,2)

err = torch.zeros([len(ranks),1])

j=0
for i in ranks:
    print("rank", i)
    core, factors = tucker(W, rank = [i,i,i], n_iter_max=10, init='svd') 
    W1 = tl.tucker_to_tensor(core, factors)
    err[j] = torch.norm((W - W1), p = 'fro')/nrm
    print(err[j])
    j =j+1

torch.save(err, 'errors3.pt')