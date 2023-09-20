# Prep pre-trained model for transfer learning 
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
import torch.optim as optim
from cnn_architecture import weights_init

class TransferLearningModel(nn.Module):

  def __init__(self, pre_trained_model, freeze_weights=False):
    super().__init__()
    self.pre_trained_model = pre_trained_model
    self.freeze_weights = freeze_weights
    self._reformat_model()

    # reinitialize weights
    self.pre_trained_model.classify[-1] = nn.Linear(
      self.pre_trained_model.classify[-1].in_features,
       self.pre_trained_model.classify[-1].out_features) 
    
    DEVICE = torch.device("cuda")
    self.pre_trained_model.to(DEVICE)

  def _reformat_model(self):
      # Freeze all the weights
      if self.freeze_weights:
        for params in self.pre_trained_model.parameters():
          params.requires_grads = False
      

  def forward(self, x):
    return self.pre_trained_model.forward(x)



class DomainAdaptionModel(nn.Module):
  def __init__(self, pre_trained_model, freeze_weights=False, num_of_layers=1):
    super().__init__()
    self.pre_trained_model = pre_trained_model
    self.freeze_weights = freeze_weights
    self._reformat_model()

    # reinitialize weights
    for i in range(1, num_of_layers + 1):
      self.pre_trained_model.classify[-i] = nn.Linear(
        self.pre_trained_model.classify[-i].in_features,
          self.pre_trained_model.classify[-i].out_features)
    
    DEVICE = torch.device("cuda")
    self.pre_trained_model.to(DEVICE)
  
  def _reformat_model(self):
      # Freeze all the weights
      if self.freeze_weights:
        for params in self.pre_trained_model.parameters():
          params.requires_grads = False
      

  def forward(self, x):
    return self.pre_trained_model.forward(x)
  