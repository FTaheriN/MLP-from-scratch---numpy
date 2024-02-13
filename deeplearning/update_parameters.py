import numpy as np

def update_params(gradients, nn_params, nn_arch, momentum, lr):#, dw_prev, db_prev):
    alpha = 0.9 * momentum
    for l, layer in enumerate(nn_arch):
        nn_params['w_change'+str(l+1)] = lr * gradients["dw" + str(l+1)] + alpha*nn_params['w_change'+str(l+1)]
        nn_params['b_change'+str(l+1)] = lr * gradients["db" + str(l+1)] + alpha*nn_params['b_change'+str(l+1)]        
        nn_params['w' + str(l+1)] -= nn_params['w_change'+str(l+1)]
        nn_params['b' + str(l+1)] -= nn_params['b_change'+str(l+1)]
    return nn_params