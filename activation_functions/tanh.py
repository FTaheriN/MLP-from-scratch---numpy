import numpy as np

def tanh(z):
  return np.tanh(z)

def d_tanh(da, z):
  return da*((1 - tanh(z))*tanh(z))