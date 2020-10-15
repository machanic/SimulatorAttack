#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.kernel_size // 2)

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            hidden = Variable(torch.zeros(size_h).cuda())
        if input is None:
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            input = Variable(torch.zeros(size_h).cuda())
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = self.dropout(f.sigmoid(rt))
        update_gate = self.dropout(f.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


