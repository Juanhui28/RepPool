import torch
import util
import numpy as np
from visualize import visualize
from my_visualize import my_visualize
import pickle
# file_name = 'save/agcn_softmax^global_adj_norm^none_COLLAB/sample000_epoch000.vis'
# file_name = 'save/sample001_epoch010.npy'

file_name = 'save/NCI1/0.25/0.002/1/Mysample000_epoch230.npy'
# file_name = 'save/COLLAB/Mysample002_epoch001.npy'
# file_name = 'save/attpool/NCI1/sample001_epoch010.npy'
# a=torch.load(file_name)

#a=np.load(file_name)

with open(file_name,'rb') as fp:
    a = pickle.load(fp)

# visualize(*a)
my_visualize(*a)

