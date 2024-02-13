import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path, train_ratio=0.7):
    df = pd.read_csv(data_path)
    y = df.iloc[:, -1]
    df = df.drop(['cityCode','made'],axis=1).join(pd.get_dummies(df.made))
    x = df.drop('price', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    x_train_min = x_train.min()
    x_train_max = x_train.max()
    x_train = (x_train-x_train_min) / (x_train_max-x_train_min)
    x_test = (x_test-x_train_min) / (x_train_max-x_train_min)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
