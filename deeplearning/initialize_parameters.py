import numpy as np

def init_layers(nn_arch, init_values, seed=12):
    np.random.seed(seed)
    # number_of_layers = len(nn_arch)
    nn_params = {}
    mu = init_values['mu']
    sigma = init_values['sigma']
    bias = init_values['bias']

    for l, layer in enumerate(nn_arch):
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        nn_params['w' + str(l+1)] = np.random.normal(mu, sigma,
            size = (layer_input_size, layer_output_size)) * 0.01
        # nn_params['w' + str(l+1)] = np.zeros((layer_input_size, layer_output_size)) #1.4
        nn_params['b' + str(l+1)] = np.zeros((1, layer_output_size))
        nn_params['w_change'+str(l+1)] = 0.0
        nn_params['b_change'+str(l+1)] = 0.0
        
    return nn_params