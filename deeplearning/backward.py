import numpy as np
import activation_functions as act_functions
import losses

def backward_switch(activation):
    act_func = {
    "sigmoid": act_functions.d_sigmoid,
    "tanh": act_functions.d_tanh,
    "relu": act_functions.d_relu,
    "leaky_relu": act_functions.d_leaky_relu,
    "linear": act_functions.d_linear,
    }
    # if not found, use relu
    return act_func.get(activation, act_functions.d_relu)

def single_layer_backward(dA, a_prev, z, w, activation):
    activation_function = backward_switch(activation)
    dz = activation_function(z, dA)
    # print("dA: ", dA.shape)
    # print("a_prev: ", a_prev.shape)
    # print("z: ", z.shape)
    # print("w: ", w.shape)
    # print("activation: ", activation)
    # print("dz: ", dz.shape)
    # print(dz)
    da_prev = dz @ w.T
    dw = a_prev.T @ dz
    db = np.sum(dz, axis=0, keepdims=True)
    return da_prev, dw, db

def backward(y, y_hat, loss, nn_params, nn_arch, cache, reg):
    gradients = {}
    if loss == "mse":
        da_prev = losses.mse_grad(y, y_hat)
        # print("FFFFFF; ", da_prev.shape)
        ll = 1
    else :
        L = len(nn_arch)-1
        z = cache['Z' + str(L+1)]
        a_prev = cache['A' + str(L)]
        w = nn_params['w' + str(L+1)]
        dz = losses.d_softmax_cross_entropy(y, y_hat)
        da_prev = dz @ w.T
        dw = a_prev.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        gradients['dw' + str(L+1)] = dw + ((reg/y_hat.shape[0]) * nn_params['w'+str(L+1)])
        gradients['db' + str(L+1)] = db
        ll = 2

    for l in range(len(nn_arch)-ll, -1, -1):
        da_curr = da_prev
        activation = nn_arch[l]['act_func']
        a_prev = cache['A' + str(l)]
        z = cache['Z' + str(l+1)]
        w = nn_params['w' + str(l+1)]
        da_prev, dw, db = single_layer_backward(da_curr, a_prev, z, w, activation)
        gradients['dw' + str(l+1)] = dw + ((reg/y_hat.shape[0]) * nn_params['w'+str(l+1)])
        gradients['db' + str(l+1)] = db
        
    return gradients