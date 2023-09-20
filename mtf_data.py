import pandas as pd
import numpy as np
import pyts
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams['font.size'] = 16

def hello():
  print("Successfully Accessed")
  return

class MTF_Generator:

  # Bins - Used to specify the number of equally spaced brackets used for the y-axis
  # Time_Window - How many data points are we inputting to the MTF
  # Interval - How do we move from one time_window to the other

  def __init__(self, n_bins=11, time_window=500, interval=500, device='cuda') -> None:
    self._raw_data = None
    self.params = {
        "n_bins": n_bins,
        "time_window": time_window,
        "interval": interval
    }
    self.device = device

    self.mtf = MarkovTransitionField(n_bins=n_bins, strategy='uniform')

  def get_params(self):
    return self.params
    
  def raw_data(self, raw_data):
    if torch.is_tensor(self._raw_data):
      self._raw_data = raw_data.to(self.device)
    else:
      self._raw_data = raw_data
    return None
    
  def process_data(self):
    params = self.get_params()
    # Determine number of matrices
    amount = (self._raw_data.shape[0] - params["time_window"])//params["interval"] + 1

    if torch.is_tensor(self._raw_data):
      samples = torch.zeros((amount, params["time_window"]))
    else:
      samples = np.zeros((amount, params["time_window"]))
    # Process Markov Transition Fields
    for i in range(samples.shape[0]):
      shift = i * params["interval"]
      samples[i] = self._raw_data[0 + shift:params["time_window"] + shift]

    return samples
    
  def create_fields(self):
    if torch.is_tensor(self._raw_data):
      var = torch.from_numpy(self.mtf.fit_transform(self.process_data()))
      var = var.to(self.device)
    else:
      var = self.mtf.fit_transform(self.process_data())

    return var

class Generate_Mtf_Images(object):

  # Input: Markov Transtion Field generated for a given dimension (default here is 224)
  # Output: Saved img of the form (Height, Width, Channels(RGB)), i.e. (224x224x3)

  def __init__(self, vmin = 0., vmax = 1., drive_path = None) -> None:
    self.params = {
      "vmin": vmin,
      "vmax": vmax,
      "path": drive_path
    }

  # Function to get the Params defined during initialisation
  def get_params(self):
    return self.params

  # Function to generate MTF images and save it in the provided folder in Google Drive
  def img_generator(self, mtf_fields, path):
      params = self.get_params()
      # Loop over to iterate over all the MTF's generated should be (224x224x3) - Output Img
      for num in range(mtf_fields.shape[0]):
        img = mtf_fields[num]
        plt.imsave(os.path.join(params["drive_path"], 'MTF', num, '.png'), img, 
                   vmin = params["vmin"], vmax = params["vmax"])
        plt.close()
      return
