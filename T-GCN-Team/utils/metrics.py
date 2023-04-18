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
                # team_1_ave = t[j][33:53]
                # team_2_ave = t[j][53:73]
                # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                # aw = torch.cat((ow, ew, dw, iw), dim=1)
                # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                # team_1_ave_inputs = team_1_ave @ aw + ab
                # team_2_ave_inputs = team_2_ave @ aw + ab
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                # ## self attention
                # com = model.attentionLayer(com)
                # output attention
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V2
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), )
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V3
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
                # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
                # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

                # oppo
                team_1_ave = t[j][33:113]
                team_2_ave = t[j][113:193]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_1_inputs = team_1_ave[:20] @ aw + ab
                team_2_1_inputs = team_2_ave[:20] @ aw + ab
                team_1_2_inputs = team_1_ave[20:40] @ aw + ab
                team_2_2_inputs = team_2_ave[20:40] @ aw + ab
                team_1_3_inputs = team_1_ave[40:60] @ aw + ab
                team_2_3_inputs = team_2_ave[40:60] @ aw + ab
                team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
                team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
                com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
                diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
                mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
                diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
                diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
                mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
                diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
                diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
                mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
                diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                ## output attention
                # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                # com = torch.cat((com1, com2), 0)
                ## co-attention
                com1 = model.attentionLayer(com1, 7, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), 12)
                com2 = model.attentionLayer(com2, 7, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), 12)
                com = torch.cat((com1, com2), 0)
                
                r_y = model.regressor1(com)
                # r_y = model.dropoutLayer1(r_y)
                r_y = model.regressor2(r_y)
                # r_y = model.dropoutLayer2(r_y)
                r_y = model.regressor3(r_y)
                # r_y = model.dropoutLayer3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y):
                    s += ((t[j][2] - r_y) ** 2)
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
                # team_1_ave = t[j][33:53]
                # team_2_ave = t[j][53:73]
                # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                # aw = torch.cat((ow, ew, dw, iw), dim=1)
                # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                # team_1_ave_inputs = team_1_ave @ aw + ab
                # team_2_ave_inputs = team_2_ave @ aw + ab
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                # ## self attention
                # com = model.attentionLayer(com)
                # output attention
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V2
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), )
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V3
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
                # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
                # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

                # oppo
                team_1_ave = t[j][33:113]
                team_2_ave = t[j][113:193]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_1_inputs = team_1_ave[:20] @ aw + ab
                team_2_1_inputs = team_2_ave[:20] @ aw + ab
                team_1_2_inputs = team_1_ave[20:40] @ aw + ab
                team_2_2_inputs = team_2_ave[20:40] @ aw + ab
                team_1_3_inputs = team_1_ave[40:60] @ aw + ab
                team_2_3_inputs = team_2_ave[40:60] @ aw + ab
                team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
                team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
                com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
                diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
                mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
                diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
                diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
                mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
                diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
                diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
                mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
                diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                ## output attention
                # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                # com = torch.cat((com1, com2), 0)
                ## co-attention
                com1 = model.attentionLayer(com1, 7, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), 12)
                com2 = model.attentionLayer(com2, 7, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), 12)
                com = torch.cat((com1, com2), 0)
                
                r_y = model.regressor1(com)
                # r_y = model.dropoutLayer1(r_y)
                r_y = model.regressor2(r_y)
                # r_y = model.dropoutLayer2(r_y)
                r_y = model.regressor3(r_y)
                # r_y = model.dropoutLayer3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y):
                    s += torch.sqrt((t[j][2] - r_y) ** 2)
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
                # team_1_ave = t[j][33:53]
                # team_2_ave = t[j][53:73]
                # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                # aw = torch.cat((ow, ew, dw, iw), dim=1)
                # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                # team_1_ave_inputs = team_1_ave @ aw + ab
                # team_2_ave_inputs = team_2_ave @ aw + ab
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                # ## self attention
                # com = model.attentionLayer(com)
                # output attention
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V2
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), )
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V3
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
                # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
                # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

                # oppo
                team_1_ave = t[j][33:113]
                team_2_ave = t[j][113:193]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_1_inputs = team_1_ave[:20] @ aw + ab
                team_2_1_inputs = team_2_ave[:20] @ aw + ab
                team_1_2_inputs = team_1_ave[20:40] @ aw + ab
                team_2_2_inputs = team_2_ave[20:40] @ aw + ab
                team_1_3_inputs = team_1_ave[40:60] @ aw + ab
                team_2_3_inputs = team_2_ave[40:60] @ aw + ab
                team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
                team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
                com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
                diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
                mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
                diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
                diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
                mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
                diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
                diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
                mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
                diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                ## output attention
                # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                # com = torch.cat((com1, com2), 0)
                ## co-attention
                com1 = model.attentionLayer(com1, 7, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), 12)
                com2 = model.attentionLayer(com2, 7, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), 12)
                com = torch.cat((com1, com2), 0)
                
                r_y = model.regressor1(com)
                # r_y = model.dropoutLayer1(r_y)
                r_y = model.regressor2(r_y)
                # r_y = model.dropoutLayer2(r_y)
                r_y = model.regressor3(r_y)
                # r_y = model.dropoutLayer3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y) and (torch.abs(r_y) > threshold):
                    if t[j][2]*r_y > 0:
                        right += 1
                    # else:
                    #     print("==={}===".format(r_y))
                    #     print(t[j])
                    game += 1

    return 0 if game == 0 else right / game

def get_return(inputs, targets, model, threshold = 0, return_ = 1):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    gain = 0.0
    gain_game = 0.0
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
                # team_1_ave = t[j][33:53]
                # team_2_ave = t[j][53:73]
                # ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                # ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                # dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                # iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                # aw = torch.cat((ow, ew, dw, iw), dim=1)
                # ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                # team_1_ave_inputs = team_1_ave @ aw + ab
                # team_2_ave_inputs = team_2_ave @ aw + ab
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                # ## self attention
                # com = model.attentionLayer(com)
                # output attention
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V2
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), )
                # com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                # com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                ## output attention V3
                # diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                # diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                # mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean), 0)
                # com = model.attentionLayer(com, torch.cat((team_1_ave_inputs, team_2_ave_inputs, diff1, diff2, mul), 0))
                # com = torch.cat((com, team_1_ave_inputs, team_2_ave_inputs), 0)

                # oppo
                team_1_ave = t[j][33:113]
                team_2_ave = t[j][113:193]
                ow = model.mask_aspect(20, model.lt1.weight, [2, 5, 8, 9, 12], 16)
                ew = model.mask_aspect(20, model.lt2.weight, [10, 14, 15], 16)
                dw = model.mask_aspect(20, model.lt3.weight, [13, 17], 16)
                iw = model.mask_aspect(20, model.lt4.weight, [19], 16)
                aw = torch.cat((ow, ew, dw, iw), dim=1)
                ab = torch.cat((model.lt1.bias, model.lt2.bias, model.lt3.bias, model.lt4.bias), dim=0)
                team_1_1_inputs = team_1_ave[:20] @ aw + ab
                team_2_1_inputs = team_2_ave[:20] @ aw + ab
                team_1_2_inputs = team_1_ave[20:40] @ aw + ab
                team_2_2_inputs = team_2_ave[20:40] @ aw + ab
                team_1_3_inputs = team_1_ave[40:60] @ aw + ab
                team_2_3_inputs = team_2_ave[40:60] @ aw + ab
                team_1_ave_inputs = team_1_ave[60:80] @ aw + ab
                team_2_ave_inputs = team_2_ave[60:80] @ aw + ab
                com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                diff1_1 = torch.sub(team_1_1_inputs, team_2_1_inputs)
                diff2_1 = torch.sub(team_2_1_inputs, team_1_1_inputs)
                mul_1 = torch.mul(team_1_1_inputs, team_2_1_inputs)
                diff1_2 = torch.sub(team_1_2_inputs, team_2_2_inputs)
                diff2_2 = torch.sub(team_2_2_inputs, team_1_2_inputs)
                mul_2 = torch.mul(team_1_2_inputs, team_2_2_inputs)
                diff1_3 = torch.sub(team_1_3_inputs, team_2_3_inputs)
                diff2_3 = torch.sub(team_2_3_inputs, team_1_3_inputs)
                mul_3 = torch.mul(team_1_3_inputs, team_2_3_inputs)
                diff1_ave = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                diff2_ave = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                mul_ave = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                ## output attention
                # com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                # com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                # com = torch.cat((com1, com2), 0)
                ## co-attention
                com1 = model.attentionLayer(com1, 7, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), 12)
                com2 = model.attentionLayer(com2, 7, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), 12)
                com = torch.cat((com1, com2), 0)
                
                r_y = model.regressor1(com)
                # r_y = model.dropoutLayer1(r_y)
                r_y = model.regressor2(r_y)
                # r_y = model.dropoutLayer2(r_y)
                r_y = model.regressor3(r_y)
                # r_y = model.dropoutLayer3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y) and (torch.abs(r_y) > threshold):
                    if t[j][2]*r_y > 0:
                        if r_y > 0:
                            gain += (return_ * t[j][193])
                        else:
                            gain += (return_ * t[j][194])
                        gain_game += 1
                            
                    game += 1

    print("===gain===", gain_game, "===game===", game)
    return 0 if game == 0 else gain / game

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