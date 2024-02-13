import numpy as np
import random
import matplotlib.pyplot as plt

def plot_images(x_train, y_train, label_names, seed, num_pics_perr_class=10):
  random.seed(seed)
  fig, axs = plt.subplots(10,num_pics_perr_class, figsize=(8,10), sharey=True)
  for i in range(0,10): #class
    for j in range(0,num_pics_perr_class): #within class
      indeces = [k for k,label in enumerate(y_train) if label==i]
      subx = x_train[indeces]
      indx = random.randrange(subx.shape[0])
      axs[i,j].imshow(np.transpose(subx[indx], (1, 2, 0)))
      axs[i,j].axes.xaxis.set_ticklabels([])
      axs[i,j].axes.yaxis.set_ticklabels([])
  for ax, row in zip(axs[:,0], label_names):
    ax.set_ylabel(row, size=8)
  fig.tight_layout()
  spt = fig.suptitle("Smaple images from CIFAR-10", fontsize="large")
  spt.set_y(1.001)
  plt.show()