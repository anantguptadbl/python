# IMPORTS
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

data=np.random.randint(2,size=(1000,10)).astype(np.float32)
a=scipy.sparse.coo_matrix(data)
i=np.concatenate((a.row.reshape(-1,1),a.col.reshape(-1,1)),axis=1).T
v=a.data
i=torch.LongTensor(i)
v=torch.FloatTensor(v)
torch.sparse.FloatTensor(i, v)
