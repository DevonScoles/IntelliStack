import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
"""
This file will contain data for an AI that can recognize if the input is Spanish or English
"""

data = [ ("me gusta comer en la cafeteria".split(), "SPANISH"),
         ("Give it to me".split(), "ENGLISH"),
         ("No creo que sea una buena idea".split(), "SPANISH"),
         ("No it is not a good idea to get lost at sea".split(), "ENGLISH") ]

test_data = [ ("Yo creo que si".split(), "SPANISH"),
              ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data :
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2



class BoWClassifier(nn.Module): #inheriting from nn.Module
    def __init__(self, num_labels, vocab_size):
        # calls init func of nn.module
        # always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # define Params needed, A and b,
        # these are the parameters of the affine mapping.
        # Torch definese nn.linear
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        # pass the input through the linear layer
        # then pass that through log_softmax
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=0)
    

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)


# the model knows it's parameters. The first output below is A and the second b
# Whenever you assign a component to a class variable in the __init__ function of a module,
# which was done with the line:
# self.linear = nn.Linear(...)
# Then through some Python magic from Pytorch devs, your module (in this case, BoWClassifier)
# will store knowledge of the nn.Linear's parameters
# print([param for param in model.parameters()])
    
# To run the model, pass in a BoW vector, but wrapped in an autograd.Varibale

# sample = data[0]
# bow_vector = make_bow_vector(sample[0], word_to_ix)
# log_probs = model(autograd.Variable(bow_vector))
# print(log_probs)

label_to_ix = {"SPANISH" : 0, "ENGLISH" : 1}

# Run on test data before we train, just to see a before-and-after
"""
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print (log_probs)

print (next(model.parameters())[:,word_to_ix["creo"]]) # Print the matrix column corresponding to "creo"
"""

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usually you want to pass over the data several times
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances. usually somwhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for instance, label in data:
        # Step 1: Remember Pytorch accumulates gradients,
        # so they need to be cleared before each instance using the .zero_grad() function
        model.zero_grad()

        # Step 2: Make our BOW vector and wrap the target in a Variable
        # as an integer. For example, if the target is SPANISH, then wrap the integer
        # 0. The loss fucntion then knows that the 0th element of the log probabilities is
        # the log probability coresponding to SPANISH
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # Step 3: Run our forward pass
        log_probs = model(bow_vec)

        # Step 4: Copmute the loss, gradients, and update the parameters by calling
        # optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print (log_probs)

print (next(model.parameters())[:,word_to_ix["creo"]],"\n") # Print the matrix column corresponding to "creo"

lst = list(model.parameters())
print(word_to_ix)
for param in lst:
    print(param)

