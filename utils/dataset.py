import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split

random_seed = 12345


def get_data(data_path: str):
    input_data = read_csv(data_path, header=None, sep=",")
    # print(input_data.shape)
    input_data = np.array(input_data)
    return input_data[:, :36].copy(), input_data[:, 36:46].copy()


def split_data(x, y):
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.1, random_state=random_seed+43276)
    # print(x1.shape, x2.shape, y1.shape, y2.shape)
    # x2, x3, y2, y3 = train_test_split(x2, y2, test_size=0.5, random_state=random_seed+43521)
    x3, y3 = [], []
    return x1, y1, x2, y2, x3, y3


def Normalization01(data):
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    return (data - data_min) / (data_max - data_min), (data_max, data_min)
