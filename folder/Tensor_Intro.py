import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

torch.manual_seed(1)

"""
#creates a torch.Tensor obj with given data. 1D vector
V_data = [ 1. , 2. , 3. ]
V = torch.Tensor(V_data)
print(np.array(V), "\nV[0]:", np.array(V[0]), "\n")

#creates tensor matrix
M_data = [[1.,2.,3.],[4.,5.,6.]]
M = torch.Tensor(M_data)
print(np.array(M), "\n\nM[0]:", np.array(M[0]), "\n")

#Creates a 3D tensor of size 2x2x2
T_data = [[ [1., 2.], [3., 4.] ],
          [ [5., 6.], [7., 8.] ]] 
T = torch.Tensor(T_data)
print(np.array(T), "\n\nT[0]:\n", np.array(T[0]), "\n")

#Adding matricies together
x = torch.Tensor([1 ,2, 3])
y = torch.Tensor([4, 5, 6])
print(np.array(x), "\n", np.array(y))


x_1 = torch.randn(2, 3)
y_1 = torch.randn(3, 3)
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
print(np.array(x_1),"\n\n",np.array(y_1),"\n\n")
z_1 = torch.cat([x_1, y_1])
# z_2 = torch.cat([x_1, y_2])
print(np.array(z_1))
print(np.array(x+y))
"""
