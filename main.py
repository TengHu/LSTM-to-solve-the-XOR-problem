from __future__ import unicode_literals, print_function, division
from io import open
import glob
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

torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
parser = argparse.ArgumentParser(description='RNN predict parity bit')

parser.add_argument(
    '--input',
    type=str,
    default='./output',
    help='location of the data corpus')

parser.add_argument('--epochs', type=int, default=3, help='# of epochs')

args = parser.parse_args()

###############################################################################
# Load Data
###############################################################################

"""sequence -> training data """
data = {}
with open(args.input,  encoding="utf-8") as f:
    for line in f:
        sequence = len(line.rstrip())
        idx = np.zeros((1, sequence, 1), dtype=np.int64)
        for i in range(0, sequence):
            idx[0][i] = line[i]
        idx = torch.LongTensor(idx)

        if sequence in data:
            data[sequence] = torch.cat((data[sequence], idx), dim=0)
        else:
            data[sequence] = idx



###############################################################################
# Build the model
###############################################################################
hidden_size = 2
model = models.simpleRNN(1, hidden_size).to(device)

###############################################################################
# Training
###############################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss().to(device)

start = time.time()
all_losses = []



def train(data):
    length = data.shape[1]  # number of chars per batch
    losses = []
    total_loss = 0
    hiddens = model.initHidden(layer=3, batch_size=args.batch_size)

    for batch_idx, idx in enumerate(range(0, length - args.bptt, args.bptt)):
        inputs, targets = get_batch(data, idx)
        detach(hiddens)

        optimizer.zero_grad()

        outputs, hiddens = model(inputs, hiddens)
        loss = get_loss(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.print_every == 0 and batch_idx > 0:
            print(
                "Epoch : {} / {}, Iteration {} / {}, Loss every {} iteration :  {}, Takes {} Seconds".
                format(epoch, args.epochs, batch_idx,
                       int((length - args.bptt) / args.bptt), args.print_every,
                       loss.item(),
                       time.time() - start))

        if batch_idx % args.plot_every == 0 and batch_idx > 0:
            losses.append(total_loss / args.plot_every)
            total_loss = 0

        if batch_idx % args.sample_every == 0 and batch_idx > 0:
            sample(warm_up_text)

        if batch_idx % args.save_every == 0 and batch_idx > 0:
            save_checkpoint({
                'epoch': epoch,
                'iter': batch_idx,
                'losses': losses,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "checkpoint_{}_epoch_{}_iteration_{}.{}.pth".format(
                int(time.time()), epoch, batch_idx, model_type))
        del loss, outputs

    return losses



def get_shuffled_data(data):
    pass
    #return train_data, valid_data



try:
    for epoch in range(1, args.epochs + 1):
        train_data, valid_data = get_shuffled_data(data)
           
except KeyboardInterrupt:
    print('#' * 90)
    print('Exiting from training early')

print('#' * 90)
print("Training finished ! Takes {} seconds ".format(time.time() - start))
