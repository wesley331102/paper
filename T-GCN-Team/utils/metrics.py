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
            if t[j][0] != 0 or t[j][1] != 0:
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

def get_rmse_name(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    s = 0.0
    num = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                team_1_list = t[j][3:18]
                team_2_list = t[j][18:33]
                team_1_list = [ elem for elem in team_1_list if elem != -1]
                team_2_list = [ elem for elem in team_2_list if elem != -1]
                team_1_list = [ int(i+30) for i in team_1_list]
                team_2_list = [ int(i+30) for i in team_2_list]
                if len(team_1_list) < 7 or len(team_2_list) < 7:
                    continue
                team_1_list_st = team_1_list[:5]
                team_1_list_b = team_1_list[5:]
                team_2_list_st = team_2_list[:5]
                team_2_list_b = team_2_list[5:]
                st11 = inp[team_1_list_st[0]]
                st12 = inp[team_1_list_st[1]]
                st13 = inp[team_1_list_st[2]]
                st14 = inp[team_1_list_st[3]]
                st15 = inp[team_1_list_st[4]]
                st21 = inp[team_2_list_st[0]]
                st22 = inp[team_2_list_st[1]]
                st23 = inp[team_2_list_st[2]]
                st24 = inp[team_2_list_st[3]]
                st25 = inp[team_2_list_st[4]]
                team_1_mean = torch.mean(inp[team_1_list_b], dim=0)
                team_2_mean = torch.mean(inp[team_2_list_b], dim=0)
                # delete
                team_1_ave = t[j][33:53]
                team_2_ave = t[j][53:73]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_ave_inputs = team_1_ave @ aw + ab
                team_2_ave_inputs = team_2_ave @ aw + ab
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                real_y = model.regressor1(com)
                real_y = model.dropoutLayer1(real_y)
                real_y = model.regressor2(real_y)
                real_y = model.dropoutLayer2(real_y)
                real_y = model.regressor3(real_y)
                real_y = model.dropoutLayer3(real_y)
                real_y = model.regressor4(real_y)
                if False in torch.isnan(real_y):
                    s += ((t[j][2] - real_y) ** 2)
                    num += 1

    return (s / num), (s / num) ** 0.5

def get_rmse_T2T(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    rmse_loss = 0.0

    for i in range(leng):
        real_y = model.regressor1(inputs[i])
        real_y = model.regressor2(real_y)
        rmse_loss += ((real_y - targets[i]) ** 2)

    return (rmse_loss / leng), (rmse_loss / leng) ** 0.5

def get_rmse_score(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    s = 0.0
    num = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                team_1_list = team_2_player[int(t[j][0])]
                team_2_list = team_2_player[int(t[j][1])]
                team_1_mean = torch.mean(inp[team_1_list], dim=0)
                team_2_mean = torch.mean(inp[team_2_list], dim=0)
                team_1_score = torch.tensor([t[j][2]])
                team_2_score = torch.tensor([t[j][3]])
                com1 = torch.cat((inp[int(t[j][0])], team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], team_2_mean), 0)
                real_y_1 = model.regressor(com1)
                real_y_2 = model.regressor(com2)
                s += ((real_y_1 - team_1_score) ** 2)
                s += ((real_y_2 - team_2_score) ** 2)
                num += 2

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
            if t[j][0] != 0 or t[j][1] != 0:
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

def get_mae_name(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    s = 0.0
    num = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                team_1_list = t[j][3:18]
                team_2_list = t[j][18:33]
                team_1_list = [ elem for elem in team_1_list if elem != -1]
                team_2_list = [ elem for elem in team_2_list if elem != -1]
                team_1_list = [ int(i+30) for i in team_1_list]
                team_2_list = [ int(i+30) for i in team_2_list]
                if len(team_1_list) < 7 or len(team_2_list) < 7:
                    continue
                team_1_list_st = team_1_list[:5]
                team_1_list_b = team_1_list[5:]
                team_2_list_st = team_2_list[:5]
                team_2_list_b = team_2_list[5:]
                st11 = inp[team_1_list_st[0]]
                st12 = inp[team_1_list_st[1]]
                st13 = inp[team_1_list_st[2]]
                st14 = inp[team_1_list_st[3]]
                st15 = inp[team_1_list_st[4]]
                st21 = inp[team_2_list_st[0]]
                st22 = inp[team_2_list_st[1]]
                st23 = inp[team_2_list_st[2]]
                st24 = inp[team_2_list_st[3]]
                st25 = inp[team_2_list_st[4]]
                team_1_mean = torch.mean(inp[team_1_list_b], dim=0)
                team_2_mean = torch.mean(inp[team_2_list_b], dim=0)
                # delete
                team_1_ave = t[j][33:53]
                team_2_ave = t[j][53:73]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_ave_inputs = team_1_ave @ aw + ab
                team_2_ave_inputs = team_2_ave @ aw + ab
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                real_y = model.regressor1(com)
                real_y = model.dropoutLayer1(real_y)
                real_y = model.regressor2(real_y)
                real_y = model.dropoutLayer2(real_y)
                real_y = model.regressor3(real_y)
                real_y = model.dropoutLayer3(real_y)
                real_y = model.regressor4(real_y)
                if False in torch.isnan(real_y):
                    s += torch.sqrt((t[j][2] - real_y) ** 2)
                    num += 1

    return (s / num)

def get_mae_T2T(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    rmse_loss = 0.0

    for i in range(leng):
        real_y = model.regressor1(inputs[i])
        real_y = model.regressor2(real_y)
        rmse_loss += torch.sqrt((real_y - targets[i]) ** 2)

    return (rmse_loss / leng)

def get_mae_score(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    s = 0.0
    num = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                team_1_list = team_2_player[int(t[j][0])]
                team_2_list = team_2_player[int(t[j][1])]
                team_1_mean = torch.mean(inp[team_1_list], dim=0)
                team_2_mean = torch.mean(inp[team_2_list], dim=0)
                team_1_score = torch.tensor([t[j][2]])
                team_2_score = torch.tensor([t[j][3]])
                com1 = torch.cat((inp[int(t[j][0])], team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], team_2_mean), 0)
                real_y_1 = model.regressor(com1)
                real_y_2 = model.regressor(com2)
                s += torch.sqrt((real_y_1 - team_1_score) ** 2)
                s += torch.sqrt((real_y_2 - team_2_score) ** 2)
                num += 2

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
            if t[j][0] != 0 or t[j][1] != 0:
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

def get_accuracy_name(inputs, targets, model, threshold = 0):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    right = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                team_1_list = t[j][3:18]
                team_2_list = t[j][18:33]
                team_1_list = [ elem for elem in team_1_list if elem != -1]
                team_2_list = [ elem for elem in team_2_list if elem != -1]
                team_1_list = [ int(i+30) for i in team_1_list]
                team_2_list = [ int(i+30) for i in team_2_list]
                if len(team_1_list) < 7 or len(team_2_list) < 7:
                    continue
                team_1_list_st = team_1_list[:5]
                team_1_list_b = team_1_list[5:]
                team_2_list_st = team_2_list[:5]
                team_2_list_b = team_2_list[5:]
                st11 = inp[team_1_list_st[0]]
                st12 = inp[team_1_list_st[1]]
                st13 = inp[team_1_list_st[2]]
                st14 = inp[team_1_list_st[3]]
                st15 = inp[team_1_list_st[4]]
                st21 = inp[team_2_list_st[0]]
                st22 = inp[team_2_list_st[1]]
                st23 = inp[team_2_list_st[2]]
                st24 = inp[team_2_list_st[3]]
                st25 = inp[team_2_list_st[4]]
                team_1_mean = torch.mean(inp[team_1_list_b], dim=0)
                team_2_mean = torch.mean(inp[team_2_list_b], dim=0)
                # delete
                team_1_ave = t[j][33:53]
                team_2_ave = t[j][53:73]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_ave_inputs = team_1_ave @ aw + ab
                team_2_ave_inputs = team_2_ave @ aw + ab
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                real_y = model.regressor1(com)
                real_y = model.dropoutLayer1(real_y)
                real_y = model.regressor2(real_y)
                real_y = model.dropoutLayer2(real_y)
                real_y = model.regressor3(real_y)
                real_y = model.dropoutLayer3(real_y)
                real_y = model.regressor4(real_y)
                if False in torch.isnan(real_y) and (torch.abs(real_y) > threshold):
                    if t[j][2]*real_y > 0:
                        right += 1
                    game += 1

    return 0 if game == 0 else right / game

def get_accuracy_T2T(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    right = 0.0
    
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    right = 0.0

    for i in range(leng):
        real_y = model.regressor1(inputs[i])
        real_y = model.regressor2(real_y)
        if targets[i] * real_y > 0:
            right += 1

    return right / leng