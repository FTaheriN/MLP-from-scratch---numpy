import numpy as np
import activation_functions as act_functions


def forward_switch(activation):
    act_func = {
    "sigmoid": act_functions.sigmoid,
    "tanh": act_functions.tanh,
    "relu": act_functions.relu,
    "leaky_relu": act_functions.leaky_relu,
    "softmax": act_functions.softmax,
    "linear": act_functions.linear,
    }
    # if not found, use relu
    return act_func.get(activation, act_functions.relu)

def single_layer_forward(w, a, b, activation):
    z = np.dot(a, w) + b
    activation_func = forward_switch(activation)
    return activation_func(z), z

def forward(x_train, nn_arch, nn_params):
    A = x_train
    cache = {}
    cache['A0'] = A
    for l, layer in enumerate(nn_arch):
        input = cache['A' + str(l)]
        w = nn_params['w' + str(l+1)]
        b = nn_params['b' + str(l+1)]
        act_func = layer['act_func']
        A, Z = single_layer_forward(w, input, b, act_func)
        cache['A' + str(l+1)] = A
        cache['Z' + str(l+1)] = Z
    return A, cache