import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_history, valid_loss_history, epochs=50):
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(np.arange(0,epochs), loss_history, label='train loss')
    ax1.plot(np.arange(0,epochs), valid_loss_history, label='validation loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss of model per epoch')
    ax1.legend()
    plt.show()
    return

def plot_cost(cost_history, epochs=50):
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(np.arange(0,epochs), cost_history)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('cost')
    ax2.set_title('Cost of model per epoch')
    plt.show()
    return
    
def plot_loss_accuracy(train_history, valid_history, epochs):
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].plot(np.arange(0,epochs), 1-np.array(train_history), label='train loss')
    ax[0].plot(np.arange(0,epochs), 1-np.array(valid_history), label='validation loss')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Model Loss per Epoch')
    ax[0].legend()

    ax[1].plot(np.arange(0,epochs), train_history, label='train accuracy')
    ax[1].plot(np.arange(0,epochs), valid_history, label='validation accuracy')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('accuracy')
    ax[1].set_title('Model Accuracy per Epoch')
    ax[1].legend()
    fig.tight_layout()
    plt.show()
    return
