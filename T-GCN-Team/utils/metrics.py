import torch
import numpy as np

def accuracy(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)

def get_mean(m, y):
    s = 0.0
    num = 0.0
    for i in range(len(y)):
        for j in range(len(y[i][0])):
            if y[i][0][j][2] != 0:
                s += ((y[i][0][j][2] - m) ** 2)
                num += 1
    
    return (s / num), (s / num) ** 0.5

def nba_metrics(p, y, o, threshold=False):
    assert len(p) == len(y) and len(p) == len(o)
    rmse_loss = 0.0
    mae_loss = 0.0
    acc = 0.0
    gain = 0.0
    game = 0.0
    for i in range(len(p)):
        rmse_loss += ((p[i] - y[i]) ** 2)
        mae_loss += torch.sqrt((p[i] - y[i]) ** 2)
        if p[i]*y[i] > 0:
            acc += 1
            gain += o[i]
        game += 1
    rmse_loss = rmse_loss / game
    rmse_loss = torch.sqrt(rmse_loss)
    mae_loss = mae_loss / game
    acc = acc / game
    gain = gain / game

    if threshold:
        for t in range(5):
            print("==========threshold: {}==========".format(str(t+1)))
            acc_th = 0.0
            gain_th = 0.0
            game_th = 0.0
            for i in range(len(p)):
                if abs(p[i]) > t+1:
                    if p[i]*y[i] > 0:
                        acc_th += 1
                        gain_th += o[i]
                    game_th += 1
            acc_th = acc_th / game_th if game_th > 0 else 0
            gain_th = gain_th / game_th if game_th > 0 else 0
            print("==========acc: {}==========".format(str(acc_th)))
            print("==========gain: {}==========".format(str(gain_th)))

    return rmse_loss, mae_loss, acc, gain