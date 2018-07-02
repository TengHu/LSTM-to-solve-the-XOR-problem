import torch
import argparse
import models
import pdb
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='RNN predict parity bit')

parser.add_argument(
    '--import_model',
    type=str,
    default="NONE",
    help='import model')
args = parser.parse_args()


feature_size = 2
hidden_size = 2
model = models.simpleLSTM(feature_size, hidden_size).to(device)
model.load_state_dict(torch.load("simple_lstm.pth"))


# visualize every weight

def print_model(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

input = "0001"
sequence = len(input)-1
ids = np.zeros((sequence, 2), dtype=np.int64)
for i in range(0, sequence):
    ids[i][int(input[i])] = 1
ids = torch.Tensor(ids).unsqueeze_(0)
               
target_id = torch.LongTensor(np.zeros((1), dtype=np.int64))
target_id[0] = int(input[-1]) # Use Tensor constructor

hiddens = model.initHidden(layer=1, batch_size=1)
            
o, hs = model(ids, hiddens)
output_prob = o[:, -1, :].exp()
id = torch.multinomial(output_prob[0], num_samples=1).item()
        
pdb.set_trace()
print (h)
