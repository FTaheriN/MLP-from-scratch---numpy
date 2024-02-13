import numpy as np
import losses

from utils import create_mini_batch_housing
from utils import get_mae

from deeplearning.f_test import test
from deeplearning.initialize_parameters import init_layers
from deeplearning.forward import forward
from deeplearning.backward import backward
from deeplearning.update_parameters import update_params




def r_train(x_train, y_train, x_valid, y_valid, nn_arch,  loss_type, weights, seed, problem, epochs=50, \
          learning_rate=0.001, momentum=False, reg=0.0, batch_size=32):
    print(x_train.shape)
    print(y_train.shape)
    nn_params = init_layers(nn_arch, weights, seed)
    num_layers = len(nn_arch)
    cost_history = []
    loss_history = []
    valid_loss_history = []

    for i in range(epochs):
        cost_history_batch = []
        loss_history_batch = []
        print("\nepoch: ", i+1)
        mini_batches = create_mini_batch_housing(x_train, y_train, batch_size)

        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch
            # print(x_mini.shape)
            # print(y_mini.shape)
            y_hat, cache = forward(x_mini, nn_arch, nn_params)
            cost = losses.mse_cost(y_mini, y_hat)
            cost_history_batch.append(cost)

            gradients = backward(y_mini, y_hat, loss_type, nn_params, nn_arch, cache, reg)
            nn_params = update_params(gradients, nn_params, nn_arch, momentum, \
                                                        learning_rate)
            
        pred, valid_loss = test(x_valid, y_valid, nn_arch, nn_params, batch_size)    
        print("cost: ", np.sum(cost_history_batch)/len(mini_batches))     
        print("Validation loss: ", valid_loss)
        # loss_history.append(np.sum(loss_history_batch)/len(mini_batches))
        cost_history.append(np.sum(cost_history_batch)/len(mini_batches))
        valid_loss_history.append(valid_loss)
        
    return nn_params, cost_history, cost_history, valid_loss_history