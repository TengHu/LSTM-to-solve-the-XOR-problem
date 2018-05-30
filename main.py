from io import open
import torch
import torch.nn.functional as F
import torchvision.models as models
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import argparse
import models
import time
import pdb
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
parser = argparse.ArgumentParser(description='RNN predict parity bit')
parser.add_argument(
    '--print_every',
    type=int,
    default=5,
    help='print every iteration')

parser.add_argument(
    '--batch_size',
    type=int,
    default=50,
    help='training batch size')

parser.add_argument(
    '--thresh',
    type=float,
    default=0.5,
    help='threshold for classification')

parser.add_argument(
    '--train',
    type=str,
    default='./train.txt',
    help='location of the training corpus')

parser.add_argument(
    '--valid',
    type=str,
    default='./valid.txt',
    help='location of the validation corpus')

parser.add_argument(
    '--test',
    type=str,
    default='./test.txt',
    help='location of the testing corpus')

parser.add_argument('--epochs', type=int, default=3, help='# of epochs')

args = parser.parse_args()

###############################################################################
# Load Data
###############################################################################
def get_data(path):
    """
    return a mapping from sequence -> 
        (# of samples, sequence, 1)
    and a mapping from sequence -> 
        (# of samples, 1)
    """
    data = {}
    targets = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            sequence = len(line)-1
            ids = np.zeros((sequence, 2), dtype=np.int64)

            for i in range(0, sequence):
                ids[i][int(line[i])] = 1
            
            target_id = torch.LongTensor(np.zeros((1), dtype=np.int64))
            target_id[0] = int(line[-1]) # Use Tensor constructor
                
            ids = torch.Tensor(ids).unsqueeze_(0).to(device)
            target_id.unsqueeze_(0).to(device)

            if sequence in data:
                data[sequence] = torch.cat((data[sequence], ids), dim=0)
                targets[sequence] = torch.cat((targets[sequence], target_id), dim=0)
            else:
                data[sequence] = ids
                targets[sequence] = target_id
    return data, targets

train_data, train_targets = get_data(args.train)
test_data, test_targets = get_data(args.test)

###############################################################################
# Build the model
##############################################################################

feature_size = 2
hidden_size = 2
model = models.simpleLSTM(feature_size, hidden_size).to(device)

###############################################################################
# Training
##############################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss().to(device)

start = time.time()

def detach(layers):
    '''
    Remove variables' parent node after each sequence, 
    basically no where to propagate gradient 
    '''
    if (type(layers) is list) or (type(layers) is tuple):
        for l in layers:
            detach(l)
    else:
        layers.detach_()

        
def evaluate(features, targets):
    cnt = 0;
    correct = 0;
    for seq in features:
        test_data = torch.utils.data.TensorDataset(features[seq], targets[seq])
        test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True, drop_last=True)
        for i, data in enumerate(test_loader, 0):
            input, targets = data
            hiddens = model.initHidden(layer=1, batch_size=args.batch_size)
            output, _ = model(input, hiddens)
            cnt += input.shape[0]
            output_prob = output[:, -1, :].exp()
            for j in range(0, args.batch_size):
                id = torch.multinomial(output_prob[j], num_samples=1).item()
                correct += (id == targets[j].numpy()[0])

    pdb.set_trace()
    return correct / cnt

def train(features, targets):
    train_data = torch.utils.data.TensorDataset(features, targets)
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    total_loss = 0
    loss = 0
        
    for i, data in enumerate(train_loader, 0):
        input, targets = data
        optimizer.zero_grad()
        hiddens = model.initHidden(layer=1, batch_size=args.batch_size)
        output, hiddens = model(input, hiddens)
        loss = criterion(output[:, -1, :], targets.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % args.print_every == 0 and i > 0:
            print(
                "Epoch : {} / {}, Iteration {} / {}, Loss every {} iteration :  {}, Takes {} Seconds".
                format(epoch, args.epochs, i, features.shape[0] / args.batch_size, args.print_every,
                       loss.item(),
                       time.time() - start))
        del loss, output
    return total_loss

try:
    for epoch in range(1, args.epochs + 1):
        for seq in train_data:
            loss = train(train_data[seq], train_targets[seq])
            print("Epoch {} Finished,  Total loss : {}\n".format(epoch, loss))
            
except KeyboardInterrupt:
    print('#' * 90)
    print('Exiting from training early')

print("Testing finished ! Accuracy : {} % ".format(evaluate(test_data, test_targets) * 100))

print('#' * 90)
print("Training finished ! Takes {} seconds ".format(time.time() - start))
