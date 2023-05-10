import pandas as pd
import numpy as np
import torch
import csv
from torch import nn



class TpaModel(nn.Module):
    def __init__(self, input_size, hidden_size, k, output_size):
        super(TpaModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=11, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=3)
        self.fc3 = nn.Linear(in_features=k+3, out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        ht = x[:, -1, :]
        hw = x[:, :-1, :]
        hw = torch.transpose(hw, 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = torch.reshape(ht, (len(ht), 3, 1))
        a = torch.bmm(Hn, ht)
        a = torch.sigmoid(a)
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)
        ht = torch.reshape(ht, (len(ht), 1, 3))
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx)
        y = torch.squeeze(y, dim=2)


        return y