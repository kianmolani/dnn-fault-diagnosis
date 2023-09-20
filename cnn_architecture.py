import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
import torch.optim as optim

import numpy as np
import random
import statistics
import time
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
import torch.optim as optim

import os
import numpy as np
import random
import statistics
import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torchvision.models as models
from torchvision import transforms

  

class CNN_Architecture(nn.Module):
  # Initialize convolution architecture backbone - handwritten
  # Conv_layer --> ReLU --> Conv_layer --> ReLU --> MaxPool --> FC layer --> FC Layer --> softmax
  def __init__(self, 
              input_dims = (3, 224, 224),
              inp_channels = 3,
              out_channels = 64,
              kernel_size = 3,
              stride = 1,
              padding = 1,
              pool_kernel = 2,
              pool_stride = 2,
              hidden_dim = 120,
              num_classes = 2,
              dtype = torch.float,
              device = 'cpu'):

    super(CNN_Architecture, self).__init__()
    
    # Input to First Convolution Layer
    C, H, W = input_dims[0], input_dims[1], input_dims[2]
    
    out_ch1 = out_channels
    out_ch2 = 128
    out_ch3 = 256
    out_ch4 = 512

    self.features = nn.Sequential(
        nn.Conv2d(in_channels= C, out_channels= out_ch1, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch1),
        nn.Conv2d(in_channels=out_ch1, out_channels=out_ch2, kernel_size=kernel_size, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch2),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(in_channels=out_ch2, out_channels=out_ch3, kernel_size=kernel_size, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch3),
        nn.Conv2d(in_channels=out_ch3, out_channels=out_ch4, kernel_size=kernel_size, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch4),
        nn.MaxPool2d(2, 2)
        )

    HH, WW = sequential_output_size(self.features,
                                    H,
                                    W,
                                    kernel_size,
                                    stride,
                                    padding,
                                    pool_kernel,
                                    pool_stride)
    

    self.input_fc_dim = int(HH*WW*out_ch4)

    self.classify = nn.Sequential(
      nn.Linear(self.input_fc_dim, hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.2),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.2),
      nn.Linear(hidden_dim, num_classes)
      )

    # apply the weight inits
    self.apply(weights_init)

  # Forward Function
  def forward(self, Input):
    # input img
    x = self.features(Input)

    x = torch.flatten(x, 1)

    output = self.classify(x)

    return output


## Utils for NN model
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def sequential_output_size(sequential, H, W, kernel_size, stride, padding, pool_kernel, pool_stride):
    for name in sequential:
      if isinstance(name, nn.Conv2d):
        H = ((H + 2*padding - kernel_size)//stride) + 1
        W = ((W + 2*padding - kernel_size)//stride) + 1
      elif isinstance(name, nn.MaxPool2d):
        H = ((H - pool_kernel)/pool_stride) + 1
        W = ((W - pool_kernel)/pool_stride) + 1
    
    return H, W



