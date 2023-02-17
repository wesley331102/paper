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

def get_rmse(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    s = 0.0
    num = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                team_1_list = team_2_player[int(t[j][0])]
                team_2_list = team_2_player[int(t[j][1])]
                team_1_mean = torch.mean(inp[team_1_list], dim=0)
                team_2_mean = torch.mean(inp[team_2_list], dim=0)
                win_rate = torch.tensor([t[j][3]])
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, win_rate), 0)
                real_y = model.regressor1(com)
                real_y = model.regressor2(real_y)
                s += ((t[j][2] - real_y) ** 2)
                num += 1

    return (s / num), (s / num) ** 0.5

def get_mae(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    s = 0.0
    num = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                team_1_list = team_2_player[int(t[j][0])]
                team_2_list = team_2_player[int(t[j][1])]
                team_1_mean = torch.mean(inp[team_1_list], dim=0)
                team_2_mean = torch.mean(inp[team_2_list], dim=0)
                win_rate = torch.tensor([t[j][3]])
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, win_rate), 0)
                real_y = model.regressor1(com)
                real_y = model.regressor2(real_y)
                s += torch.sqrt((t[j][2] - real_y) ** 2)
                num += 1

    return (s / num)

def get_accuracy(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    right = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                team_1_list = team_2_player[int(t[j][0])]
                team_2_list = team_2_player[int(t[j][1])]
                team_1_mean = torch.mean(inp[team_1_list], dim=0)
                team_2_mean = torch.mean(inp[team_2_list], dim=0)
                win_rate = torch.tensor([t[j][3]])
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, win_rate), 0)
                real_y = model.regressor1(com)
                real_y = model.regressor2(real_y)
                if t[j][2]*real_y > 0:
                    right += 1
                game += 1

    right = right / game
    return right