import numpy as np

def leaky_relu(z, alpha=0.01):
  return np.maximum(alpha*z, z)

def d_leaky_relu(da, z, alpha=0.01):
  return da*(np.where(z>0, 1, alpha)) 