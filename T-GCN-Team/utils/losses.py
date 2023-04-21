import torch
import math
import numpy as np

def nba_mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    mse_loss = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                real_y = torch.tanh(model.regressor(com))*model.feat_max_val
                mse_loss += ((real_y - t[j][2]) ** 2)
                game += 1

    mse_loss = mse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss

    return mse_loss + reg_loss

def nba_rmse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                real_y = model.regressor(com)
                rmse_loss += torch.sqrt((real_y - t[j][2]) ** 2)
                game += 1

    rmse_loss = rmse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_rmse_with_player_with_regularizer_loss(inputs, targets, model, team_2_player, lamda=1.5e-3, using_other: bool=False):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0

    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
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
                rmse_loss += ((real_y - t[j][2]) ** 2)
                game += 1

    rmse_loss = rmse_loss / game
    rmse_loss = torch.sqrt(rmse_loss)
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_mae_with_player_with_regularizer_loss(inputs, targets, model, team_2_player, lamda=1.5e-3, using_other: bool=False):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0

    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
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
                rmse_loss += torch.sqrt((real_y - t[j][2]) ** 2)
                game += 1

    rmse_loss = rmse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_mae_with_player_with_regularizer_loss_name(inputs, targets, model, lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0
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
                # ## self attention
                # com = model.attentionLayer(com)
                # output attention
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V2
                diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), torch.cat((team_1_ave_inputs, diff1, mul), 0))
                com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V3
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
                # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
                # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

                # oppo
                # team_1_ave = t[j][33:113]
                # team_2_ave = t[j][113:193]
                # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                # aw = torch.cat((ow, ew, dw, iw), dim=1)
                # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                # team_1_1_inputs = team_1_ave[:20] @ aw + ab
                # team_2_1_inputs = team_2_ave[:20] @ aw + ab
                # team_1_2_inputs = team_1_ave[20:40] @ aw + ab
                # team_2_2_inputs = team_2_ave[20:40] @ aw + ab
                # team_1_3_inputs = team_1_ave[40:60] @ aw + ab
                # team_2_3_inputs = team_2_ave[40:60] @ aw + ab
                # team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
                # team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
                # com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                # com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                # diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
                # diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
                # mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
                # diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
                # diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
                # mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
                # diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
                # diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
                # mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
                # diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                ## output attention
                # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                # com = torch.cat((com1, com2), 0)
                ## co-attention
                # com1, ave1 = model.attentionLayer(com1, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0))
                # com2, ave2 = model.attentionLayer(com2, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0))
                # com = torch.flatten(torch.cat((com1, com2, ave1, ave2), 0))

                r_y = model.regressor1(com)
                # r_y = model.dropoutLayer1(r_y)
                r_y = model.regressor2(r_y)
                # r_y = model.dropoutLayer2(r_y)
                r_y = model.regressor3(r_y)
                # r_y = model.dropoutLayer3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y):
                    rmse_loss += torch.sqrt((r_y - t[j][2]) ** 2)
                    game += 1
    rmse_loss = rmse_loss / game
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_mae_with_player_with_regularizer_loss_T2T(inputs, targets, model, lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    rmse_loss = 0.0

    for i in range(leng):
        real_y = model.regressor1(inputs[i])
        real_y = model.regressor2(real_y)
        rmse_loss += torch.sqrt((real_y - targets[i]) ** 2)

    rmse_loss = rmse_loss / leng
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return rmse_loss + reg_loss

def nba_rmse_with_score_loss(inputs, targets, model, team_2_player, lamda=1.5e-3, isMean=True):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    rmse_loss = 0.0

    if isMean:
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
                    rmse_loss += ((real_y_1 - team_1_score) ** 2)
                    rmse_loss += ((real_y_2 - team_2_score) ** 2)
                    game += 2

    rmse_loss = rmse_loss / game
    rmse_loss = torch.sqrt(rmse_loss)
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
            if t[j][0] != 0 or t[j][1] != 0:
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

# def nba_cross_entropy_with_player_name(inputs, targets, model, isValid=False):
#     assert inputs.shape[0] == targets.shape[0]
#     leng = inputs.shape[0]
#     acc = 0.0
#     game = 0.0
#     ce = 0.0
#     p = []
#     y = []
#     for i in range(leng):
#         inp = inputs[i]
#         t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
#         for j in range(t.shape[0]):
#             if t[j][0] != 0 or t[j][1] != 0:
#                 team_1_list = t[j][3:18]
#                 team_2_list = t[j][18:33]
#                 team_1_list = [ elem for elem in team_1_list if elem != -1]
#                 team_2_list = [ elem for elem in team_2_list if elem != -1]
#                 team_1_list = [ int(i+30) for i in team_1_list]
#                 team_2_list = [ int(i+30) for i in team_2_list]
#                 if len(team_1_list) < 7 or len(team_2_list) < 7:
#                     continue
#                 team_1_list_st = team_1_list[:5]
#                 team_1_list_b = team_1_list[5:]
#                 team_2_list_st = team_2_list[:5]
#                 team_2_list_b = team_2_list[5:]
#                 st11 = inp[team_1_list_st[0]]
#                 st12 = inp[team_1_list_st[1]]
#                 st13 = inp[team_1_list_st[2]]
#                 st14 = inp[team_1_list_st[3]]
#                 st15 = inp[team_1_list_st[4]]
#                 st21 = inp[team_2_list_st[0]]
#                 st22 = inp[team_2_list_st[1]]
#                 st23 = inp[team_2_list_st[2]]
#                 st24 = inp[team_2_list_st[3]]
#                 st25 = inp[team_2_list_st[4]]
#                 team_1_mean = torch.mean(inp[team_1_list_b], dim=0)
#                 team_2_mean = torch.mean(inp[team_2_list_b], dim=0)
#                 # delete
#                 team_1_ave = t[j][33:53]
#                 team_2_ave = t[j][53:73]
#                 ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
#                 ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
#                 dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
#                 iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
#                 aw = torch.cat((ow, ew, dw, iw), dim=1)
#                 ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
#                 team_1_ave_inputs = team_1_ave @ aw + ab
#                 team_2_ave_inputs = team_2_ave @ aw + ab
#                 com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
#                 # ## self attention
#                 # com = model.attentionLayer(com)
#                 # output attention
#                 # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
#                 # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
#                 # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
#                 ## output attention V2
#                 diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
#                 diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
#                 mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
#                 com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), torch.cat((team_1_ave_inputs, diff1, mul), 0))
#                 com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
#                 com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
#                 ## output attention V3
#                 # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
#                 # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
#                 # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
#                 # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
#                 # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
#                 # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

#                 # oppo
#                 # team_1_ave = t[j][33:113]
#                 # team_2_ave = t[j][113:193]
#                 # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
#                 # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
#                 # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
#                 # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
#                 # aw = torch.cat((ow, ew, dw, iw), dim=1)
#                 # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
#                 # team_1_1_inputs = team_1_ave[:20] @ aw + ab
#                 # team_2_1_inputs = team_2_ave[:20] @ aw + ab
#                 # team_1_2_inputs = team_1_ave[20:40] @ aw + ab
#                 # team_2_2_inputs = team_2_ave[20:40] @ aw + ab
#                 # team_1_3_inputs = team_1_ave[40:60] @ aw + ab
#                 # team_2_3_inputs = team_2_ave[40:60] @ aw + ab
#                 # team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
#                 # team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
#                 # com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
#                 # com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
#                 # diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
#                 # diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
#                 # mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
#                 # diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
#                 # diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
#                 # mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
#                 # diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
#                 # diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
#                 # mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
#                 # diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
#                 # diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
#                 # mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
#                 ## output attention
#                 # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
#                 # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
#                 # com = torch.cat((com1, com2), 0)
#                 ## co-attention
#                 # com1, ave1 = model.attentionLayer(com1, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0))
#                 # com2, ave2 = model.attentionLayer(com2, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0))
#                 # com = torch.flatten(torch.cat((com1, com2, ave1, ave2), 0))

#                 r_y = model.regressor1(com)
#                 # r_y = model.dropoutLayer1(r_y)
#                 r_y = model.regressor2(r_y)
#                 # r_y = model.dropoutLayer2(r_y)
#                 r_y = model.regressor3(r_y)
#                 # r_y = model.dropoutLayer3(r_y)
#                 r_y = model.regressor4(r_y)
#                 if False in torch.isnan(r_y):
#                     r_y_c = model.sig(r_y)
#                     p.append(r_y_c)
#                     if t[j][2] > 0:
#                         y.append(1.0)
#                         if r_y_c > 0.5:
#                             acc += 1
#                     else:
#                         y.append(0.0)
#                         if r_y_c < 0.5:
#                             acc += 1
#                     game += 1
#     res = model.loss_F(torch.tensor(p), torch.tensor(y))
#     if isValid:
#         print(acc/game)
#         print("==============")
#         print(res)
#     return res

# def nba_cross_entropy_output(inputs, targets, model, lamda=1.5e-3):
#     assert inputs.shape[0] == targets.shape[0]
#     leng = inputs.shape[0]
#     a = []
#     b = []
#     for i in range(leng):
#         inp = inputs[i]
#         t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
#         for j in range(t.shape[0]):
#             if t[j][0] != 0 or t[j][1] != 0:
#                 team_1_list = t[j][3:18]
#                 team_2_list = t[j][18:33]
#                 team_1_list = [ elem for elem in team_1_list if elem != -1]
#                 team_2_list = [ elem for elem in team_2_list if elem != -1]
#                 team_1_list = [ int(i+30) for i in team_1_list]
#                 team_2_list = [ int(i+30) for i in team_2_list]
#                 if len(team_1_list) < 7 or len(team_2_list) < 7:
#                     continue
#                 team_1_list_st = team_1_list[:5]
#                 team_1_list_b = team_1_list[5:]
#                 team_2_list_st = team_2_list[:5]
#                 team_2_list_b = team_2_list[5:]
#                 st11 = inp[team_1_list_st[0]]
#                 st12 = inp[team_1_list_st[1]]
#                 st13 = inp[team_1_list_st[2]]
#                 st14 = inp[team_1_list_st[3]]
#                 st15 = inp[team_1_list_st[4]]
#                 st21 = inp[team_2_list_st[0]]
#                 st22 = inp[team_2_list_st[1]]
#                 st23 = inp[team_2_list_st[2]]
#                 st24 = inp[team_2_list_st[3]]
#                 st25 = inp[team_2_list_st[4]]
#                 team_1_mean = torch.mean(inp[team_1_list_b], dim=0)
#                 team_2_mean = torch.mean(inp[team_2_list_b], dim=0)
#                 # delete
#                 team_1_ave = t[j][33:53]
#                 team_2_ave = t[j][53:73]
#                 ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
#                 ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
#                 dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
#                 iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
#                 aw = torch.cat((ow, ew, dw, iw), dim=1)
#                 ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
#                 team_1_ave_inputs = team_1_ave @ aw + ab
#                 team_2_ave_inputs = team_2_ave @ aw + ab
#                 com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
#                 # ## self attention
#                 # com = model.attentionLayer(com)
#                 # output attention
#                 # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
#                 # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
#                 # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
#                 ## output attention V2
#                 diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
#                 diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
#                 mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
#                 com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), torch.cat((team_1_ave_inputs, diff1, mul), 0))
#                 com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
#                 com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
#                 ## output attention V3
#                 # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
#                 # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
#                 # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
#                 # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
#                 # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
#                 # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

#                 # oppo
#                 # team_1_ave = t[j][33:113]
#                 # team_2_ave = t[j][113:193]
#                 # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
#                 # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
#                 # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
#                 # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
#                 # aw = torch.cat((ow, ew, dw, iw), dim=1)
#                 # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
#                 # team_1_1_inputs = team_1_ave[:20] @ aw + ab
#                 # team_2_1_inputs = team_2_ave[:20] @ aw + ab
#                 # team_1_2_inputs = team_1_ave[20:40] @ aw + ab
#                 # team_2_2_inputs = team_2_ave[20:40] @ aw + ab
#                 # team_1_3_inputs = team_1_ave[40:60] @ aw + ab
#                 # team_2_3_inputs = team_2_ave[40:60] @ aw + ab
#                 # team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
#                 # team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
#                 # com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
#                 # com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
#                 # diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
#                 # diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
#                 # mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
#                 # diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
#                 # diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
#                 # mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
#                 # diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
#                 # diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
#                 # mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
#                 # diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
#                 # diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
#                 # mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
#                 ## output attention
#                 # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
#                 # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
#                 # com = torch.cat((com1, com2), 0)
#                 ## co-attention
#                 # com1, ave1 = model.attentionLayer(com1, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0))
#                 # com2, ave2 = model.attentionLayer(com2, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0))
#                 # com = torch.flatten(torch.cat((com1, com2, ave1, ave2), 0))

#                 r_y = model.regressor1(com)
#                 # r_y = model.dropoutLayer1(r_y)
#                 r_y = model.regressor2(r_y)
#                 # r_y = model.dropoutLayer2(r_y)
#                 r_y = model.regressor3(r_y)
#                 # r_y = model.dropoutLayer3(r_y)
#                 r_y = model.regressor4(r_y)
#                 if False in torch.isnan(r_y):
#                     m = torch.nn.Softmax()
#                     r_y = m(r_y)
#                     a.append(r_y.detach().numpy()[0])
#                     b.append(t[j][2])
        
#     return a, b

def nba_output(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                p.append(model.regressor(com))
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
                if t[j][0] != 0 or t[j][1] != 0:
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
                    y.append(t[j][2])
    else:
        for i in range(leng):
            inp = inputs[i]
            t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
            for j in range(t.shape[0]):
                if t[j][0] != 0 or t[j][1] != 0:
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
                    y.append(t[j][2])
    return p, y

def nba_output_with_player_name(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
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
                # ## self attention
                # com = model.attentionLayer(com)
                # output attention
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V2
                diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), torch.cat((team_1_ave_inputs, diff1, mul), 0))
                com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V3
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
                # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
                # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

                # oppo
                # team_1_ave = t[j][33:113]
                # team_2_ave = t[j][113:193]
                # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                # aw = torch.cat((ow, ew, dw, iw), dim=1)
                # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                # team_1_1_inputs = team_1_ave[:20] @ aw + ab
                # team_2_1_inputs = team_2_ave[:20] @ aw + ab
                # team_1_2_inputs = team_1_ave[20:40] @ aw + ab
                # team_2_2_inputs = team_2_ave[20:40] @ aw + ab
                # team_1_3_inputs = team_1_ave[40:60] @ aw + ab
                # team_2_3_inputs = team_2_ave[40:60] @ aw + ab
                # team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
                # team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
                # com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                # com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                # diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
                # diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
                # mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
                # diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
                # diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
                # mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
                # diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
                # diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
                # mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
                # diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                ## output attention
                # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                # com = torch.cat((com1, com2), 0)
                ## co-attention
                # com1, ave1 = model.attentionLayer(com1, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0))
                # com2, ave2 = model.attentionLayer(com2, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0))
                # com = torch.flatten(torch.cat((com1, com2, ave1, ave2), 0))
                
                r_y = model.regressor1(com)
                # r_y = model.dropoutLayer1(r_y)
                r_y = model.regressor2(r_y)
                # r_y = model.dropoutLayer2(r_y)
                r_y = model.regressor3(r_y)
                # r_y = model.dropoutLayer3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y):
                    p.append(r_y)
                    y.append(t[j][2])
    return p, y

def nba_output_with_player_T2T(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()

    for i in range(leng):
        real_y = model.regressor1(inputs[i])
        real_y = model.regressor2(real_y)
        p.append(real_y)
        y.append(targets[i])

    return p, y

def nba_output_with_player_score(inputs, targets, model, team_2_player):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                team_1_list = team_2_player[int(t[j][0])]
                team_2_list = team_2_player[int(t[j][1])]
                team_1_mean = torch.mean(inp[team_1_list], dim=0)
                team_2_mean = torch.mean(inp[team_2_list], dim=0)
                com1 = torch.cat((inp[int(t[j][0])], team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], team_2_mean), 0)
                p.append(model.regressor(com1))
                y.append(t[j][2])
                p.append(model.regressor(com2))
                y.append(t[j][3])
    return p, y

def nba_rmse_output(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    p, y = list(), list()
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                p.append(model.regressor(com))
                y.append(t[j][2])
    return p, y