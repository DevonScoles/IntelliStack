# DEEP LEARNING BUILDING BLOCKS: AFFINE MAPS, NON-LINEARITIES, and OBJECTIVES
import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

# AFFINE MAPS
"""
#affine map is a function of f(x)
# f(x) = Ax + b
# from a matrix A and vectors x,b. b is refferenced to as the bias term

# Pytorch and other deep learning frameworks 
# map by row input instead of columns like linear algebra

lin = nn.Linear(5, 3)
data = autograd.Variable( torch.randn(2,5) ) # data is 2x5, A maps from 5 - 3
# can we map "data" under A?

print(lin(data)) #yes
"""

# NON-LINEARTIIES
"""
# with affine maps f(x) = Ax+b and g(x) = Cx+d what is f(g(x))?
#   f(g(x)) = A(Cx+d)+b = AC+(Ad+b)
#   AC is a matrix and Ad+b is a vector
#   composing affine maps gives us an affine map
#   using non-linearities between the affine layers we can make a much more powerful model

data = autograd.Variable(torch.randn(2, 2))
print(data, "\n")
print(F.relu(data))
"""

# SOFTMAX AND PROBABILITIES
"""
#Softmax(x) is non-linearity which outputs probability distribution 
# where each element is non-negative and sum over all components is 1
#   exp(x_i) / Sigma_j (exp(x_j))
#   where x is vector of numbers (any kind)

# Softmax is also in torch.functional
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())
print(F.log_softmax(data, dim=0))
"""

# Objective Functions(loss function or cost function)
"""
# loss function is calculated by choosing a training instance and running it through the NN
#   then computing the loss of the output.
#   The parameters of the model are updated via taking the derivative of the loss function
#   confidently WRONG answers from the model will result in HIGH loss
#   confidently RIGHT answers will result in LOW loss

# loss functions like negative lof likelihood loss are common objective
#   for multi-class classification.
"""

#OPTIMIZATION AND TRAINING
"""
# Variable's konw how to compute gradients with respect to the 
# things (Tensors) that were used to compute it
#since our loss is an autograd.Variable() we can compute the gradients

# Using the vanilla Standard Gradient Descient algorithm is okay but using
#  different algorithms for this such as Adam or RMSProp will boost performance
"""

