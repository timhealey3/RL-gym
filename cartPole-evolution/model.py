import numpy as np
import torch

'''
To make recombination and mutation easier, we create populition of 1D tensors that we must unpack into indiivual params for each layer
'''

def model(x, unpacked_params):
    l1,b1,l2,b2,l3,b3 = unpacked_params
    y = torch.nn.functional.linear(x,l1,b1) 
    y = torch.relu(y)
    y = torch.nn.functional.linear(y,l2,b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y,l3,b3)
    y = torch.log_softmax(y,dim=0)
    return y