import numpy as np
import losses

from utils import create_mini_batch_cifar
from utils import get_accuracy_value, get_mae

from deeplearning.f_test import test
from deeplearning.initialize_parameters import init_layers
from deeplearning.forward import forward
from deeplearning.backward import backward
from deeplearning.update_parameters import update_params




def train(x_train, y_train, x_valid, y_valid, nn_arch,  loss, weights, seed, problem, epochs=50, \
          learning_rate=0.001, momentum=False, reg=0.0, batch_size=32):
    print(x_train.shape)
    print(y_train.shape)
    nn_params = init_layers(nn_arch, weights, seed)
    num_layers = len(nn_arch)
    cost_history = []
    accuracy_history = []
    valid_accuracy_history = []

    for i in range(epochs):
        cost_history_batch = []
        accuracy_history_batch = []
        print("\nepoch: ", i+1)
        mini_batches = create_mini_batch_cifar(x_train, y_train, batch_size)

        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch
            # print(x_mini.shape)
            # print(y_mini.shape)
            y_hat, cache = forward(x_mini, nn_arch, nn_params)

            if loss == "mse":
                cost = losses.mse_cost(y_mini, y_hat)
                cost_history_batch.append(cost)
            else:
                cost =  losses.crossentropy_cost(y_mini, y_hat, nn_params, num_layers, reg)
                cost_history_batch.append(cost)

            accuracy = get_accuracy_value(y_mini, y_hat)
            accuracy_history_batch.append(accuracy)

            gradients = backward(y_mini, y_hat, loss, nn_params, nn_arch, cache, reg)
            nn_params = update_params(gradients, nn_params, nn_arch, momentum, \
                                                        learning_rate)
            
        pred, valid_accuracy = test(x_valid, y_valid, nn_arch, nn_params, batch_size)    
        print("cost: ", np.sum(cost_history_batch)/len(mini_batches))        
        print("Train Accuracy: ", np.sum(accuracy_history_batch)/len(mini_batches))
        print("Validation Accuracy: ", valid_accuracy)
        accuracy_history.append(np.sum(accuracy_history_batch)/len(mini_batches))
        cost_history.append(np.sum(cost_history_batch)/len(mini_batches))
        valid_accuracy_history.append(valid_accuracy)
        
    return nn_params, cost_history, accuracy_history, valid_accuracy_history