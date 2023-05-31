#Computation Graphs and Automatic Differentiation
import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

# VARIABLES AND BACKPROPAGATION FUNCITON
"""
#Variables wrap tensor objects
x = autograd.Variable( torch.Tensor([1,2,3]), requires_grad=True)
#to access data use .data
# print(x.data)

#Variables can do same operations as Tensors
y = autograd.Variable( torch.Tensor([4,5,6]), requires_grad=True)
z = x + y
# print(z)

# print(z.grad_fn

s = z.sum()
# print(s,z)
# print(s.grad_fn)
# derrivative of sum with respect to first component of x:
# ds / dx_0
# s knows it was created from the sum of z and z knows it was created from the sum of x + y so:
# s = z_0{x_0 + y_0} + z_1{x_1 + y_1} + z_2{x_2 + y_2}


# using .backward() on any variable will run backpropagation starting from it
s.backward()
print(x.grad)
"""

# WHAT NOT TO DO IN ORDER TO COMPUTER BACKPROP

#understanding the following is pertinent to understanding deep learning
x = torch.randn((2,2))
y = torch.randn((2,2))
z = x + y # because these are random Tensor types, backprop is not possible

var_x = autograd.Variable(x)
var_y = autograd.Variable(y)
var_z = var_x + var_y
#var_z contains enough info to compute gradients
print("var_z.grad_fn", var_z.grad_fn)
var_z_data = var_z.data #gets wrapped Tensor object out of var_z
new_var_z = autograd.Variable( var_z_data ) #rewrap Tensor in new Variable

# new Variable still DOES NOT have information to backprop to x and y
print("new var_z.grad_fn",new_var_z.grad_fn)
# because we broke the variable chain when we ripped the tensor out of var_z using .data
# and created new_var_z we have broken the variable chain and 
# the loss Variable we create wont be able to realize the Variable exists