import numpy as np

def regularization_cost(nn_params, layers, reg):
    reg_term = np.sum(np.square(nn_params['w1']))
    for l in range(2,layers+1):
        reg_term += np.sum(np.square(nn_params['w'+str(l)]))
    return (reg / 2) * reg_term

def mse_cost(y, y_hat):
    m = y_hat.shape[0]
    cost = np.sum(np.power(y - y_hat, 2)) / m
    return np.squeeze(cost)

def mse_grad(y, y_hat):
    return -2*(y - y_hat) / y_hat.shape[0]

def crossentropy_cost(y, y_hat, nn_params, layers, reg=0.01):
    # print(y_hat[np.arange(y_hat.shape[0]), np.argmax(y, axis=1)])
    correct_logprobs = -np.log(y_hat[np.arange(y_hat.shape[0]), np.argmax(y, axis=1)])
    if reg != 0.0:
        reg_term = regularization_cost(nn_params, layers, reg)
        return np.squeeze((np.sum(correct_logprobs) + reg_term) / y_hat.shape[0])
    
    return np.sum(correct_logprobs) / y_hat.shape[0]

def d_softmax_cross_entropy(y, y_hat):
    dZ = y_hat.copy()
    dZ[np.arange(y_hat.shape[0]),np.argmax(y, axis=1)] -= 1
    dZ /= y_hat.shape[0]
    return dZ