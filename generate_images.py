import matplotlib
import numpy as np

from pyts.image import MarkovTransitionField
from scipy import signal
from skimage.transform import resize
from matplotlib import pyplot as plt


class MTF_Generator:
  def __init__(self, n_bins=None, n_data_pts=None, raw_data=None) -> None:
     self._raw_data = raw_data
     self._params = {
        "n_bins": n_bins,
        "n_data_pts": n_data_pts 
     }
     self.mtf = MarkovTransitionField(n_bins=n_bins, strategy='uniform')

  def generate_mtf(self):
      
      n_data_pts = self._params["n_data_pts"]

      # derive number of samples
      n_samples = int(self._raw_data.shape[0] // n_data_pts)

      # pre-create numpy that holds data
      samples = np.zeros((n_samples, n_data_pts))

      # populate samples
      for i in range(n_samples):
        samples[i] = self._raw_data[i * n_data_pts : (i + 1) * n_data_pts].squeeze()
      
      # transform data into MTF image
      mtf = self.mtf.transform(samples)

      # transform MTF image into RGB image
      mtf_rgb = np.zeros((mtf.shape[0], mtf.shape[1], mtf.shape[2], 3))
      for i in range(mtf_rgb.shape[0]):
        sm = matplotlib.cm.ScalarMappable(cmap='rainbow')
        im = sm.to_rgba(mtf[i])
        mtf_rgb[i] = resize(im, (mtf.shape[1], mtf.shape[2]))[:, :, :3]
      
      return mtf_rgb
  

# class Spec_Generator:
#     def __init__(self, sampling_frequency=None, n_data_pts=None, raw_data=None) -> None:
#       self._raw_data = raw_data
#       self._params = {
#         "sampling_frequency": sampling_frequency,
#         "n_data_pts": n_data_pts
#      }
      
#     def generate_spec(self):

#       n_data_pts = self._params["n_data_pts"]

#       # derive number of samples
#       n_samples = int(self._raw_data.shape[0] // n_data_pts)

#       # pre-create numpy that holds data
#       samples = np.zeros((n_samples, n_data_pts))

#       # populate samples
#       for i in range(n_samples):
#         samples[i] = self._raw_data[i * n_data_pts : (i + 1) * n_data_pts].squeeze()
      
#       # compute spectrograms
#       spectrogram = np.zeros((samples.shape[0], 100, 100, 3))
#       for i in range(n_samples):
#         f, t, Sxx = signal.spectrogram(samples[i], fs = self._params["sampling_frequency"])
#         spectrogram[i] = resize(Sxx, (100, 100))
#         im = plt.pcolormesh(t, f, Sxx, shading='gouraud')

#          plt.ylabel('Frequency [Hz]')

#         plt.xlabel('Time [sec]')
#         plt.ylim([0, 5000])
#         plt.show()
         
#       # transform MTF image into RGB image
#       mtf_rgb = np.zeros((mtf.shape[0], mtf.shape[1], mtf.shape[2], 3))
#       for i in range(mtf_rgb.shape[0]):
#         sm = matplotlib.cm.ScalarMappable(cmap='rainbow')
#         im = sm.to_rgba(mtf[i])
#         mtf_rgb[i] = resize(im, (mtf.shape[1], mtf.shape[2]))[:, :, :3]
      
#       return mtf_rgb
