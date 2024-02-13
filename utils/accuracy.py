import numpy as np

def get_accuracy_value(y, y_hat):
    y = y.argmax(axis=1)
    y_hat = y_hat.argmax(axis=1)
    accuracy = np.sum(y == y_hat) / y_hat.shape[0]
    return accuracy

def get_mae(y, y_hat):
    return np.sum(np.abs(y-y_hat)) / y_hat.shape[0]