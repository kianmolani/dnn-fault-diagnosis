import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class FullNN(nn.Module):
  def __init__(self, pre_trained_model):
        super().__init__()
        self.ftt = BasePreTrainNN(pre_trained_model)
        self.mtf = BasePreTrainNN(pre_trained_model)

        self.fc = nn.Linear(128, 2) 
        
        # Custom final layer
        self.custom_model = nn.Sequential(
          nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), # Need to figure out 
          nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
        )
  
  def forward(self, fft, mtf):
    x_ftt = self.ftt.forward(fft)
    x_mtf = self.ftt.forward(mtf)

    combine = x_ftt + x_mtf
    output = self.custom_model(combine)

    flat_output = torch.flatten(output)

    flat_output = self.fc(flat_output)

    return flat_output

class BasePreTrainNN(nn.Module):
    """
    Regression approximation via 3-FC NN layers.
    The network input features are one-dimensional as well as the output features.
    The network hidden sizes are 100 and 100.
    Activations are Tanh
    """
    def __init__(self, pre_trained_model, freeze_weights=False):
        super().__init__()
        # --- Your code here
        self.pre_trained_model = pre_trained_model
        self._reformat_model(freeze_weights)
        
        # The custom architecture  
        self.custom_model = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), # assume input is 512
          nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        )
        # ---
    
    def _forward_pre_trained(self, x):
      x = self.pre_trained_model.conv1(x)
      x = self.pre_trained_model.bn1(x)
      x = self.pre_trained_model.relu1(x)
      x = self.pre_trained_model.maxpool(x)

      x = self.pre_trained_model.layer1(x)
      x = self.pre_trained_model.layer2(x)
      x = self.pre_trained_model.layer3(x)
      x = self.pre_trained_model.layer4(x)
      
      return x

    def _reformat_model(self, freeze_weights):
      # Freeze all the weights
      if freeze_weights:
        for params in self.pre_trained_model.parameters():
          params.requires_grads = False
      
      # Clear the top layers so we can peform transfer learning
      self.pre_trained_model.avgpool = None
      self.pre_trained_model.fc = None

      
      return None

    def forward(self, x):

        # Pre_trained Model
        x = self._forward_pre_trained(x)

        # custom architecture
        x = self.custom_model(x)
        # ---
        return x


# This part part is not finished


# import time
# import torch.optim as optim

# num_train = 50

# # Network of CNN Model
# bn_model = CNN_Architecture(input_dims=(C, H, W), inp_channels=C, num_classes=2,
#                             dtype=torch.float32, device='cuda')

# # Optimizer SGD (Want Adam)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(bn_model.parameters(), lr=0.001, momentum= 0.9)

# NUM_EPOCHS = 4
# st_time = time.time()
# for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         # print("Inputs Shape", inputs.shape)
#         # print("Labels [should be batch size]", labels)
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = bn_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 20 == 19:    # print every 20 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
#             running_loss = 0.0

# print('Finished Training')



def train_step(model, train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0. # TODO: Modify the value
    # Initialize the train loop
    # --- Your code here
    model.train() 
    # ---
    for batch_idx, (data, target) in enumerate(train_loader):
        # --- Your code here
        print(batch_idx)
        print(data)
        print(target)
        yhat = model.forward(data)
    
        # first lets zero our gradients
        optimizer.zero_grad()
        # now we compute our loss
        mse_loss = F.mse_loss(yhat, target)
        # compute the gradients with respect to our loss
        mse_loss.backward()
        # update our parameters using the optimizer!
        optimizer.step()
        # ---
        train_loss += mse_loss.item()
    return train_loss/len(train_loader)


def val_step(model, val_loader) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    # --- Your code here
    model.eval()

    # ---
    for batch_idx, (data, target) in enumerate(val_loader):
        # loss = None
        # --- Your code here
        yhat = model(data)
        mse_loss = F.mse_loss(yhat, target)
        # ---
        val_loss += mse_loss.item()
    return val_loss/len(val_loader)


def train_model(model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = None
    # Initialize the optimizer
    # --- Your code here
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ---

    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = None
        val_loss_i = None
        # --- Your code here
        train_loss_i = train_step(model, train_dataloader, optimizer)
        val_loss_i = val_step(model, val_dataloader)
        # ---
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses

