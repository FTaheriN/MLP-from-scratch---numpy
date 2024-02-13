import numpy as np

def one_hot_labels(y):
   y_hot = np.zeros((y.shape[0], y.max() + 1))
   y_hot[np.arange(y.shape[0]), y] = 1
   return y_hot