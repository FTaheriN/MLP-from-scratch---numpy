import numpy as np

def relu(z):
    return (np.maximum(0, z))

def d_relu(z, da):
    dz = np.array(da, copy = True)
    dz[z <= 0] = 0
    return dz