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

def load_T2T_features(feat_path):
    feat_df = pickle.load(open(feat_path, "rb"))
    return feat_df
    
def load_team_player_dict(path):
    res = pickle.load(open(path, "rb"))
    return res

def load_targets(target_path):
    target_df = pickle.load(open(target_path, "rb"))
    return target_df

def load_T2T_targets(target_path):
    target_df = pickle.load(open(target_path, "rb"))
    return target_df

def load_y_max(target_path):
    max_y = -1.00
    target_df = pickle.load(open(target_path, "rb"))
    for date in target_df:
        for key in date.keys():
            if np.abs(date[key]) > max_y:
                max_y = np.abs(date[key])
    return max_y

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None, skiprows=1, index_col=0)
    adj = np.array(adj_df, dtype=dtype)
    return adj

def dict_to_list(data):
    result = list()
    for data_dict in data:
        res = list()
        for key in data_dict:
            res.append((key[0], key[1],  data_dict[key][0], data_dict[key][1]))
        while len(res) != 15:
            res.append((0, 0, 0.0, 0.0))
        result.append(res)
    return result

def dict_to_list_name(data, output_attention):
    result = list()
    for data_dict in data:
        res = list()    
        for key in data_dict:
            if output_attention in ["V2_reverse", "co"]:
                if not (len(data_dict[key][3]) == 4):
                    data_dict[key][3].insert(0, data_dict[key][3][2])
                team_1_ave = [item for sublist in data_dict[key][3] for item in sublist]
                if not (len(data_dict[key][4]) == 4):
                    data_dict[key][4].insert(0, data_dict[key][4][2])
                team_2_ave = [item for sublist in data_dict[key][4] for item in sublist]
                assert len(team_1_ave) == 80 and len(team_2_ave) == 80

            k1 = list()
            k1.extend(data_dict[key][1])
            while len(k1) != 15:
                k1.append(-1) 
            k2 = list() 
            k2.extend(data_dict[key][2])
            while len(k2) != 15:
                k2.append(-1)  
            all_ = list()
            all_.append(key[0])
            all_.append(key[1])
            all_.append(data_dict[key][0])
            all_.extend(k1)
            all_.extend(k2)
            if output_attention in ["V2_reverse", "co"]:
                all_.extend(team_1_ave)
                all_.extend(team_2_ave)
            else:
                all_.extend(data_dict[key][3])
                all_.extend(data_dict[key][4])
            # odds
            all_.append(data_dict[key][5])
            all_.append(data_dict[key][6])
            res.append(all_)
        z = list()
        z_size = 33
        if output_attention in ["V2_reverse", "co"]:
            z_size = 195
        else:
            z_size = 75
        while len(z) != z_size:
            z.append(0)
        while len(res) != 15:
            res.append(z)
        result.append(res)
    return result

def generate_dataset(
    # data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
    data, y, output_attention, split_ratio=0.8, normalize=True
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

    train_size = int(data_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:data_len]
    y = dict_to_list_name(y, output_attention)
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

def generate_dataset_T2T(
    data, y, split_ratio=0.8, normalize=True
):
    data_len = data.shape[0]
    if normalize:
        for i in range(data.shape[2]):
            m = np.max(data[:, :, i])
            data[:, :, i] = data[:, :, i] / float(m)

    train_size = int(data_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_y = y[:train_size]
    test_y = y[train_size:]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(train_size):
        train_X.append(np.array(train_data[i, :5, :]))
        train_Y.append(np.array(train_y[i, -1]))
    for i in range(data_len-train_size):
        test_X.append(np.array(test_data[i, :5, :]))
        test_Y.append(np.array(test_y[i, -1]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

def generate_torch_datasets(
    data, y, output_attention, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        y, 
        output_attention,
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

def generate_torch_datasets_T2T(
    data, y, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset_T2T(
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