import numpy as np
import pandas as pd
import torch
import pickle

def data_transform(data):
    return data

def load_features(feat_path, p_feature_path, dtype=np.float32):
    feat_df = pickle.load(open(feat_path, "rb"))
    feat_p_df = pickle.load(open(p_feature_path, "rb"))
    feat = np.array(np.concatenate((feat_df, feat_p_df), axis=1), dtype=dtype)
    return feat
    # feat_df = pd.read_csv(feat_path)
    # feat = np.array(feat_df, dtype=dtype)
    # return 
    
def load_team_player_dict(path):
    res = pickle.load(open(path, "rb"))
    return res

def load_targets(target_path):
    target_df = pickle.load(open(target_path, "rb"))
    return target_df
    # feat_df = pd.read_csv(feat_path)
    # feat = np.array(feat_df, dtype=dtype)
    # return feat

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None, skiprows=1, index_col=0)
    adj = np.array(adj_df, dtype=dtype)
    return adj

def load_adjacency_matrices(adjs_path, dtype=np.float32):
    adjs = []
    for adj_path in adjs_path:
        adj_df = pd.read_csv(adj_path, header=None, skiprows=1, index_col=0)
        adj = np.array(adj_df, dtype=dtype)
        adjs.append(adj)
    return adjs

def dict_to_list(data):
    result = list()
    for data_dict in data:
        res = list()
        for key in data_dict:
            res.append((key[0], key[1],  data_dict[key]))
        while len(res) != 15:
            res.append((0, 0, 0.0))
        result.append(res)
    return result

def generate_dataset(
    # data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
    data, y, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    # if time_len is None:
    data_len = data.shape[0]
    if normalize:
        for i in range(data.shape[2]):
            m = np.max(data[:, :, i])
            data[:, :, i] = data[:, :, i] / float(m)
        # max_val = np.max(data)
        # print(max_val)
        # data = data / max_val

    train_size = int(data_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:data_len]
    y = dict_to_list(y)
    train_y = y[:train_size]
    test_y = y[train_size:data_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    seq_len = 10
    pre_len = 1
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_y[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_y[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), train_Y, np.array(test_X), test_Y


def generate_torch_datasets(
    # data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
    data, y, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        y, 
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
