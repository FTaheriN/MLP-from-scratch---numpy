import numpy as np

def create_mini_batch_cifar(x, y, batch_size):
    num_batches = x.shape[0] // batch_size
    mini_batches = []
    data = np.hstack((x, y))
    np.random.shuffle(data)

    for i in range(num_batches):
        # print((i + 1)*batch_size)
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-10]
        Y_mini = mini_batch[:, -10:]#.reshape((-1, 10))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0], :]
        X_mini = mini_batch[:, :-10]
        Y_mini = mini_batch[:, -10:]#.reshape((-1, 10))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

def create_mini_batch_housing(x, y, batch_size):
    num_batches = x.shape[0] // batch_size
    mini_batches = []
    y = y.reshape((-1, 1))
    data = np.hstack((x, y))
    np.random.shuffle(data)

    for i in range(num_batches):
        # print((i + 1)*batch_size)
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0], :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches