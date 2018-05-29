import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class simpleLSTM(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(simpleLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        """Weights"""
        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.h2o = nn.Linear(hidden_size, feature_size)

    def forward(self, input, hiddens):
        outputs, h = self.lstm(input, hidden)
        return F.log_softmax(outputs, 2), h

    def initHidden(self, layer=1, batch_size=50, use_gpu=True):
        h = torch.randn(1, batch_size, self.hidden_size).pin_memory().to(device)
        return h
