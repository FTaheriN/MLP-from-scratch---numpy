import numpy as np

def sigmoid(z):
    sig = 1 / 1 + np.exp(-z)
    return sig

def d_sigmoid(z, da):
    sig = sigmoid(z)
    return da * sig * (1-sig)