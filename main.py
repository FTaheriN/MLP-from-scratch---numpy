import time
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

import plots
import dataloaders
from deeplearning.f_train import train
from deeplearning.r_train import r_train
from deeplearning.f_test import test
from utils import read_yaml_config

############################## Reading Model Parameters ##############################
config = read_yaml_config()
random_seed = config['random_seed']
problem = config['problem']
data_path = config['dataset1']['path'] # change to dataset2 for the regression task
weights = config['weights']
lr = config['learning_rate'][0]
epochs = config['num_epochs'][0]
batch_size = config['batch_size'][2]
model = config['model1'] # change to model3 for regression
input_type = config['input_type'][0]
momentum = config['momentum']
regularization = config['regularization']
act_func = config['act_func'][0]
last_act_func = config['last_act_func'][0]
loss = config['loss_func'][0]

#################################### Loading Data ####################################
def load_cifar():
    x_train, y_train = dataloaders.load_train_data(data_path, input_type)
    x_test, y_test = dataloaders.load_test_data(data_path, input_type)
    label_names = dataloaders.load_label_names(data_path)

    # # plot_pictures.plot_images(x_train, y_train, label_names, random_seed)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, x_test, y_train, y_test, label_names

def load_paris_housing():
    return dataloaders.load_data(data_path)

################################# Network Architecture ################################
nn_arch = []
for l in range(0, len(model['layer_dims'])-1):
    layer = {}
    layer['input_dim'] = model['layer_dims'][l]
    layer['output_dim'] = model['layer_dims'][l+1]
    layer['act_func'] = act_func
    if l == len(model['layer_dims'])-2:
        layer['act_func'] = last_act_func
    nn_arch.append(layer)
print(nn_arch)

################################### Training Process ##################################
def k_fold(x_train, x_test, y_train, y_test):
    x = np.vstack(x_test, x_train)
    y = np.vstack(y_test, y_train)

    test_acc = []

    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, valid_index in skf.split(x_train, y_train):
        x_train_fold, x_valid_fold = x_train[train_index], x_train[valid_index]
        y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]
        y_train_fold = dataloaders.one_hot_labels(y_train_fold)
        y_valid_fold = dataloaders.one_hot_labels(y_valid_fold)


        nn_params, cost, train_accuracy, valid_accuracy = train( x_train_fold,y_train_fold, \
                                                            x_valid_fold,y_valid_fold, \
                                                            nn_arch, loss, weights,\
                                                            random_seed, problem, epochs, lr, \
                                                            momentum, regularization, batch_size)
        y_hat_test, test_accuracy = test(x_valid_fold, y_valid_fold, nn_arch, nn_params, batch_size)
        test_acc.append(test_accuracy)

    print("validation accuracy: ", np.mean(test_acc))
    return 

def simple(x_train, x_test, y_train, y_test, label_names):

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, \
                                                      random_state=random_seed, stratify=y_train)
    y_train = dataloaders.one_hot_labels(y_train)
    y_valid = dataloaders.one_hot_labels(y_valid)
    y_test = dataloaders.one_hot_labels(y_test)

    start = time.time()
    nn_params, cost, train_accuracy, valid_accuracy = train(x_train,y_train, \
                                                            x_valid,y_valid, \
                                                            nn_arch, loss, weights,\
                                                            random_seed, problem, epochs, lr, \
                                                            momentum, regularization, batch_size)

    end = time.time()
    print("execution time(seconds): ", end - start)

    plots.plot_cost(cost, epochs)
    plots.plot_loss_accuracy(train_accuracy, valid_accuracy, epochs)
    y_hat_test, test_accuracy = test(x_test, y_test, nn_arch, nn_params, batch_size)
    print("\n\n")
    print("Test accuracy: ", test_accuracy)
    print("Test loss: ", 1-test_accuracy)
    print("\n\n")
    plots.plot_conf_matrix(y_test, y_hat_test, label_names)

def regression(x_train, x_test, y_train, y_test):
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, \
                                                          random_state=random_seed)
    
    start = time.time()
    nn_params, cost, train_loss, valid_loss = r_train(x_train,y_train, \
                                                            x_valid,y_valid, \
                                                            nn_arch, loss, weights,\
                                                            random_seed, problem, epochs, lr, \
                                                            momentum, regularization, batch_size)

    end = time.time()
    print("execution time(seconds): ", end - start) 

    plots.plot_cost(cost, epochs)
    plots.plot_loss(train_loss, valid_loss, epochs)
    y_hat_test, test_loss = test(x_test, y_test, nn_arch, nn_params, batch_size)
    print("\n\n")
    print("Test loss: ", np.sqrt(test_loss))
    print("\n\n")
    return 

    

def main():
    label_names = []
    if problem == "classification":
        x_train, x_test, y_train, y_test, label_names = load_cifar()
        training = 'simple'
        if training == 'k-fold':
            k_fold(x_train, x_test, y_train, y_test)
        else:
            simple(x_train, x_test, y_train, y_test, label_names)
    else:
        x_train, x_test, y_train, y_test = load_paris_housing()
        regression(x_train, x_test, y_train, y_test)
    return

main()