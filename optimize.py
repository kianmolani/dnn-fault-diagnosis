# Tools used to train and save best model
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import datasets, transforms
import os
import multiprocessing
from torchvision import datasets, transforms
from sklearn import metrics


class SaveBestModel(object):
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, model, optimizer, criterion, path,
          best_valid_loss=float('inf'), best_val_acc = 0.0):
        self.best_valid_loss = best_valid_loss
        self.best_val_acc = best_val_acc
        self.model = model
        self.optimizer = optimizer
        self.path = path
        self.criterion = criterion
        
    def best_loss(self, current_val_loss, current_val_acc, epoch):

        if current_val_loss < self.best_valid_loss and current_val_acc >= self.best_val_acc:
            self.best_valid_loss = current_val_loss
            self.best_val_acc = current_val_acc
            print(f"\nBest validation loss and acc: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            try:
              torch.save({'epoch': epoch+1, 'model_state_dict': self.model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.criterion,}, 
                  f'{self.path}/MMDclassify_Source_norm_DA_TD_Finetune_classify_layers_mixed12k.pth')
              print(f'file saved to {self.path}/MMDclassify_Source_norm_DA_TD_Finetune_classify_layers_mixed12k.pth')
            except OSError:
              print('Can not save file')


def criterion_mmd(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    XX = torch.exp(-gamma * torch.cdist(X, X) ** 2 / 2)
    YY = torch.exp(-gamma * torch.cdist(Y, Y) ** 2 / 2)
    XY = torch.exp(-gamma * torch.cdist(X, Y) ** 2 / 2)

    out = XX.mean() + YY.mean() - 2 * XY.mean()
    return out


# Training Function for Target Domain Only
def train_model_target(source_model, target_model, trainloader, optimizer, criterion, device):
    
    source_model.eval()
    target_model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    total = 0
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # print(images.shape)

        with torch.no_grad():
          # output_source = source_model(images)
          source_output = images
          count = 0
          source_feature = []

          for source_layer in source_model.features:
              source_output = source_layer(source_output)

          source_output = source_output.flatten(start_dim = 1)
          for source_layer in source_model.classify:
              source_output = source_layer(source_output)
              count += 1
              if count == 1 or count == 4:
                source_feature.append(source_output.flatten(start_dim = 1))

          source_features = torch.cat(source_feature, dim = 1)          

        target_output = images
        count = 0
        target_feature = []
        target_features = torch.tensor([0])
        for target_layer in target_model.features:
          target_output = target_layer(target_output)
        
        target_output = target_output.flatten(start_dim = 1)
        for target_layer in target_model.classify:
          target_output = target_layer(target_output)
          count += 1
          if count == 1 or count == 4:
            target_feature.append(target_output.flatten(start_dim = 1))
        
        target_features = torch.cat(target_feature, dim = 1)

        output_target = target_model(images)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + parameter updates
        loss_ce = criterion(output_target, labels)
        loss_mmd = criterion_mmd(torch.mean(source_features, dim=0), torch.mean(target_features, dim=0))
        # print(loss_mmd)
        loss = loss_ce + loss_mmd
        loss.backward()
        optimizer.step()

        # To calculate total loss over the epoch
        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(output_target.data, 1)
        total += labels.size(0)
        train_running_correct += (preds == labels).sum().item()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * (train_running_correct / total)
    
    return epoch_loss, epoch_acc

# Train model for Source Domain
def train_model(model, trainloader, optimizer, criterion, device):
    
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    total = 0
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + parameter updates
        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        # To calculate total loss over the epoch
        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(output.data, 1)
        total += labels.size(0)
        train_running_correct += (preds == labels).sum().item()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * (train_running_correct / total)
    
    return epoch_loss, epoch_acc


# Validation Function for Source and Target Domain
def val_model(model, valloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    total = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(valloader), total=len(valloader)):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # forward pass + loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate loss over the epoch
            valid_running_loss += loss.item()

            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / len(valloader)
    epoch_acc = 100. * (valid_running_correct / total)
    return epoch_loss, epoch_acc