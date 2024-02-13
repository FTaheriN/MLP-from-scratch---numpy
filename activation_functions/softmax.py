import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z/np.sum(exp_z, axis=1, keepdims=True)