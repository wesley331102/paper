import torch
import math
import numpy as np

def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / 2
    return mse_loss + reg_loss

def nba_mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    # reg_loss = 0.0
    # for param in model.parameters():
    #     reg_loss += torch.sum(param ** 2) / 2
    # reg_loss = lamda * reg_loss
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    mse_loss = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                # real_y = model.regressor(com)
                real_y = torch.tanh(model.regressor(com))*model.feat_max_val
                mse_loss += ((real_y - t[j][2]) ** 2)
                game += 1

    mse_loss = mse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    # mse_loss = torch.sum((inputs - targets) ** 2) / 2
    return mse_loss + reg_loss
    # return mse_loss

def nba_rmse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                real_y = model.regressor(com)
                # real_y = torch.tanh(model.regressor(com))*model.feat_max_val
                rmse_loss += torch.sqrt((real_y - t[j][2]) ** 2)
                game += 1

    rmse_loss = rmse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_rmse_with_player_with_regularizer_loss(inputs, targets, model, team_2_player, lamda=1.5e-3, isMean=True, using_other: bool=False):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0

    if isMean:
        for i in range(leng):
            inp = inputs[i]
            t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
            for j in range(t.shape[0]):
                if t[j][0] != 0 and t[j][1] != [0]:
                    team_1_list = team_2_player[int(t[j][0])]
                    team_2_list = team_2_player[int(t[j][1])]
                    team_1_mean = torch.mean(inp[team_1_list], dim=0)
                    team_2_mean = torch.mean(inp[team_2_list], dim=0)
                    if using_other:
                        others = torch.tensor([t[j][3], t[j][4], t[j][5], t[j][6]])
                        com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, others), 0)
                    else:
                        win_rate = torch.tensor([t[j][3]])
                        com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, win_rate), 0)
                    real_y = model.regressor1(com)
                    real_y = model.regressor2(real_y)
                    # real_y = torch.tanh(model.regressor(com))*model.feat_max_val

                    # rmse_loss += torch.sqrt((real_y - t[j][2]) ** 2)
                    # game += 1
                    rmse_loss += ((real_y - t[j][2]) ** 2)
                    game += 1
    # else:
    #     for i in range(leng):
    #         inp = inputs[i]
    #         t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
    #         for j in range(t.shape[0]):
    #             if t[j][0] != 0 and t[j][1] != [0]:
    #                 team = torch.zeros(42, 64)
    #                 team_1_list = team_2_player[int(t[j][0])]
    #                 team_2_list = team_2_player[int(t[j][1])]
    #                 y = 0
    #                 z = 21
    #                 for player in team_1_list:
    #                     team[y] = inp[player]
    #                     y += 1
    #                 team[y] = inp[int(t[j][0])]
    #                 for player in team_2_list:
    #                     team[z] = inp[player]
    #                     z += 1
    #                 team[z] = inp[int(t[j][1])]
    #                 com = torch.flatten(team)
    #                 real_y = model.regressor1(com)
    #                 real_y = model.regressor2(real_y)
    #                 # real_y = torch.tanh(model.regressor(com))*model.feat_max_val
    #                 rmse_loss += torch.sqrt((real_y - t[j][2]) ** 2)
    #                 game += 1

    rmse_loss = rmse_loss / game
    rmse_loss = torch.sqrt(rmse_loss)
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_mae_with_player_with_regularizer_loss(inputs, targets, model, team_2_player, lamda=1.5e-3, isMean=True, using_other: bool=False):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0

    if isMean:
        for i in range(leng):
            inp = inputs[i]
            t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
            for j in range(t.shape[0]):
                if t[j][0] != 0 and t[j][1] != [0]:
                    team_1_list = team_2_player[int(t[j][0])]
                    team_2_list = team_2_player[int(t[j][1])]
                    team_1_mean = torch.mean(inp[team_1_list], dim=0)
                    team_2_mean = torch.mean(inp[team_2_list], dim=0)
                    if using_other:
                        others = torch.tensor([t[j][3], t[j][4], t[j][5], t[j][6]])
                        com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, others), 0)
                    else:
                        win_rate = torch.tensor([t[j][3]])
                        com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, win_rate), 0)
                    real_y = model.regressor1(com)
                    real_y = model.regressor2(real_y)
                    # real_y = torch.tanh(model.regressor(com))*model.feat_max_val
                    rmse_loss += torch.sqrt((real_y - t[j][2]) ** 2)
                    game += 1
    rmse_loss = rmse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_cross_entropy_loss_with_player(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    ce_loss = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                if t[j][2] > 0:
                    team_1_list = team_2_player[int(t[j][0])]
                    team_2_list = team_2_player[int(t[j][1])]
                    team_1_mean = torch.mean(inp[team_1_list], dim=0)
                    team_2_mean = torch.mean(inp[team_2_list], dim=0)
                    com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean), 0)
                    real_y = model.regressor(com)
                    out = 1 / (1 + math.exp(-real_y[0].detach().numpy()))
                    ce_loss -= np.log(out)
                game += 1

    ce_loss = ce_loss / game
    return torch.tensor([ce_loss], requires_grad=True)

def nba_output(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                p.append(model.regressor(com))
                # p.append(torch.tanh(model.regressor(com))*model.feat_max_val)
                y.append(t[j][2])
    return p, y

def nba_output_with_player(inputs, targets, model, team_2_player, isMean=True, using_other: bool=False):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
    if isMean:
        for i in range(leng):
            inp = inputs[i]
            t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
            for j in range(t.shape[0]):
                if t[j][0] != 0 and t[j][1] != [0]:
                    team_1_list = team_2_player[int(t[j][0])]
                    team_2_list = team_2_player[int(t[j][1])]
                    team_1_mean = torch.mean(inp[team_1_list], dim=0)
                    team_2_mean = torch.mean(inp[team_2_list], dim=0)
                    if using_other:
                        others = torch.tensor([t[j][3], t[j][4], t[j][5], t[j][6]])
                        com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, others), 0)
                    else:
                        win_rate = torch.tensor([t[j][3]])
                        com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], team_1_mean, team_2_mean, win_rate), 0)
                    p.append(model.regressor2(model.regressor1(com)))
                    # p.append(torch.tanh(model.regressor(com))*model.feat_max_val)
                    y.append(t[j][2])
    else:
        for i in range(leng):
            inp = inputs[i]
            t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
            for j in range(t.shape[0]):
                if t[j][0] != 0 and t[j][1] != [0]:
                    team = torch.zeros(42, 64)
                    team_1_list = team_2_player[int(t[j][0])]
                    team_2_list = team_2_player[int(t[j][1])]
                    x = 0
                    z = 21
                    for player in team_1_list:
                        team[x] = inp[player]
                        x += 1
                    team[x] = inp[int(t[j][0])]
                    for player in team_2_list:
                        team[z] = inp[player]
                        z += 1
                    team[z] = inp[int(t[j][1])]
                    com = torch.flatten(team)
                    p.append(model.regressor(com))
                    # p.append(torch.tanh(model.regressor(com))*model.feat_max_val)
                    y.append(t[j][2])
    return p, y

def nba_rmse_output(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                p.append(model.regressor(com))
                # p.append(torch.tanh(model.regressor(com))*model.feat_max_val)
                y.append(t[j][2])
    return p, y