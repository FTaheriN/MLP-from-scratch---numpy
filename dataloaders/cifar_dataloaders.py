import numpy as np

# http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
   import pickle
   with open(file, 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
   return dict


data_dir = "datasets\cifar-10-batches-py"

def load_train_data(file, input_type):
   batch_dict = unpickle(file + "/data_batch_1")
   x_train = batch_dict[b'data']
   y_train = batch_dict[b'labels']
   
   for i in range(2,6):
     batch_dict = unpickle(file + "/data_batch_{}".format(i))
     x_train = np.vstack((x_train, batch_dict[b'data']))
     y_train += batch_dict[b'labels']

   x_train = x_train.reshape(len(x_train), 3, 32, 32)
   y_train = np.array(y_train)

   if input_type == "normalized":
      x_train = (x_train-x_train.min()) / x_train.max()
   elif input_type == "standardized":
      x_train = (x_train - x_train.mean(axis=(0,2,3), keepdims=True)) / x_train.std(axis=(0,2,3), keepdims=True)
   else:
      x_train = x_train
   return (x_train, y_train)

def load_test_data(file, input_type):
   batch_dict = unpickle(file + "/test_batch")
   x_test = batch_dict[b'data']
   y_test = batch_dict[b'labels']
   x_test = x_test.reshape(len(x_test), 3, 32, 32)
   y_test = np.array(y_test)

   if input_type == "normalized":
      x_test = (x_test-x_test.min()) / x_test.max()
   elif input_type == "standardized":
      x_test = (x_test - x_test.mean(axis=(0,2,3), keepdims=True)) / x_test.std(axis=(0,2,3), keepdims=True)
   else:
      x_test = x_test
   return (x_test, y_test)

def load_label_names(file=data_dir):
   meta_data_dict = unpickle(file + "/batches.meta")
   label_names = meta_data_dict[b'label_names']
   label_names = np.array(label_names)
   label_names = [str(label).split('\'')[1] for label in label_names]
   return label_names

