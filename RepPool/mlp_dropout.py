from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

# sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
# from pytorch_util import weights_init

# class MLPRegression(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(MLPRegression, self).__init__()
#
#         self.h1_weights = nn.Linear(input_size, hidden_size)
#         self.h2_weights = nn.Linear(hidden_size, 1)
#
#         weights_init(self)
#
#     def forward(self, x, y = None):
#         h1 = self.h1_weights(x)
#         h1 = F.relu(h1)
#
#         pred = self.h2_weights(h1)
#
#         if y is not None:
#             y = Variable(y)
#             mse = F.mse_loss(pred, y)
#             mae = F.l1_loss(pred, y)
#             return pred, mae, mse
#         else:
#             return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, dropout=0.0):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.dropout = dropout
        # self.with_dropout = with_dropout

        torch.nn.init.xavier_normal_(self.h1_weights.weight.t())
        torch.nn.init.constant_(self.h1_weights.bias, 0)
        torch.nn.init.xavier_normal_(self.h2_weights.weight.t())
        torch.nn.init.constant_(self.h2_weights.bias, 0)
        if self.dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        # weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.dropout > 0.001:
            h1 = self.dropout_layer(h1)

        # if self.with_dropout:
        #     h1 = F.dropout(h1, training=self.training)

        logits_tmp = self.h2_weights(h1)
        logits = F.log_softmax(logits_tmp, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits
