                                                                                                                                                                                 from __future__ import unicode_literals, print_function, division
from io import open
import glob
import torch
import pylab
import torch.nn.functional as F
import torchvision.models as models
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import unicodedata
import string
import time
import math
import matplotlib.ticker as ticker
import torch.utils.data as data_utils
import shutil
import pdb
import torch.multiprocessing as mp
import pickle

torch.manual_seed(1)

###############################################################################                                                                                                                                                                        
# Helper Functions                                                                                                                                                                                                                                     
###############################################################################                                                                                                                                                                        


def generate_binary_string(length=1):
    sum = 0
    s = ""
    for i in range(0, length):
        temp = random.randint(0, 1)
        sum += temp
        s += str(temp)
    return (s, sum % 2)


## Output: (Sequence, Batch Size, 1)                                                                                                                                                                                                                   
def string2tensor(binary_string):
    tensor = np.zeros((len(binary_string), 1, 1), dtype=np.float32)
    for i in range(len(binary_string)):
        tensor[i][0][0] = int(binary_string[i])
    return torch.Tensor(tensor)


###############################################################################                                                                                                                                                                        
# Load Data                                                                                                                                                                                                                                            
###############################################################################                                                                                                                                                                        

num_samples = 100
length = 50
for i in range(0, num_samples):
    data = torch.cat(string2tensor(generate_binary_string(length)), dim=1)

###############################################################################                                                                                                                                                                        
# Build the model                                                                                                                                                                                                                                      
###############################################################################   

class ParityGenerator(nn.Module):
    def __init__(self, batch_size, hidden_size):
        super(ParityGenerator, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size)
        self.h2o = nn.Linear(hidden_size, 2)

    def forward(self, input, hidden):
        outputs, __ = self.lstm(input, hidden)
        return self.h2o(outputs[0])

    def initHidden(self):
        return (
            Variable(
                torch.randn(
                    1, self.batch_size,
                    self.hidden_size).pin_memory().cuda(non_blocking=True)),
            Variable(
                torch.randn(
                    1, self.batch_size,
                    self.hidden_size).pin_memory().cuda(non_blocking=True)))

###############################################################################                                                                                                                                                                        
# Training                                                                                                                                                                                                                                             
###############################################################################                                                                                                                                                                        
epochs = 3
batch_size = 100
hidden_size = 200


def train(epoch_iter, model, data):
    all_losses = []
    total_loss = 0

    iters = int(data.shape[1] / batch_size)


start = time.time()

model = ParityGenerator(batch_size, hidden_size)

## shuffle data                                                                                                                                                                                                                                        
## get validation data                                                                                                                                                                                                                                 

for epoch_iter in range(0, epochs):
    # training                                                                                                                                                                                                                                         
    train(epoch_iter, model, data)

    # validation                                                                                                                                                                                                                                       
    #validate(model)                                                                                                                                                                                                                                   

print(string2tensor(generate_binary_string(3)[0]))

end = time.time()
print("Training finished ! Takes {} seconds ".format(end - start))

###############################################################################                                                                                                                                                                        
# Testing                                                                                                                                                                                                                                              
###############################################################################                                                                                                                                                                        

## generate data                               
