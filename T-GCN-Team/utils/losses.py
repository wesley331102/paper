import torch
import math
import numpy as np

def nba_loss_funtion_with_regularizer_loss(inputs, targets, model, loss_type:str="nba_mae", output_attention:str="None", lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    loss = 0.0
    p, y, o = list(), list(), list()
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
                com = torch.zeros(0)
                # delete
                if output_attention in ["self", "V1", "V2", "None", "encoder"]:
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
                    # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                    if output_attention == "self":
                        com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean, team_1_ave_inputs), 0))
                        com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean, team_2_ave_inputs), 0))
                        com = torch.flatten(torch.cat((com1, com2), 0))
                    if output_attention == "V1":
                        # output attention V1
                        com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                        com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                        com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                    elif output_attention == "V2":
                        # output attention V2
                        diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                        diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                        mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                        com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), torch.cat((team_1_ave_inputs, diff1, mul), 0))
                        com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                        com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                    elif output_attention == "encoder":
                        # com1 = model.linear_transformation(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean, team_1_ave_inputs), 0))
                        # com1 = model.tanh(com1)
                        # com1 = com1.reshape((8, model.model.hyperparameters.get("hidden_dim")))
                        # com2 = model.linear_transformation(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean, team_2_ave_inputs), 0))
                        # com2 = model.tanh(com2)
                        # com2 = com2.reshape((8, model.model.hyperparameters.get("hidden_dim")))
                        # com1 = model.transformer_encoder(com1)
                        # com2 = model.transformer_encoder(com2)
                        com1 = model.transformer_encoder(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean, team_1_ave_inputs), 0).reshape((8, model.model.hyperparameters.get("hidden_dim"))))
                        com2 = model.transformer_encoder(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean, team_2_ave_inputs), 0).reshape((8, model.model.hyperparameters.get("hidden_dim"))))
                        com1 = com1.reshape((8, model.model.hyperparameters.get("hidden_dim")))
                        com2 = com2.reshape((8, model.model.hyperparameters.get("hidden_dim")))
                        com = torch.cat((com1, com2), 0)
                        com = torch.flatten(com)
                    elif output_attention == "None":
                        com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean, team_1_ave_inputs), 0)
                        com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean, team_2_ave_inputs), 0)
                        com = torch.cat((com1, com2), 0)
                elif output_attention in ["V2_reverse", "co", "encoder_all"]:
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
                    if output_attention == "V2_reverse":
                        ## output attention reverse
                        com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                        com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                        com = torch.cat((com1, com2), 0)
                    elif output_attention == "co":
                        # output co-attention
                        com1, ave1 = model.attentionLayer(com1, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0))
                        com2, ave2 = model.attentionLayer(com2, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0))
                        com = torch.flatten(torch.cat((com1, com2, ave1, ave2), 0))
                    elif output_attention == "encoder_all":
                        # output encoder_all
                        com1 = torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0)
                        com2 = torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0)
                        com1 = model.transformer_encoder(torch.cat((com1.reshape((7, model.model.hyperparameters.get("hidden_dim"))), team_1_1_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff1_1.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_1.reshape((1, model.model.hyperparameters.get("hidden_dim"))), team_1_2_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff1_2.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_2.reshape((1, model.model.hyperparameters.get("hidden_dim"))), team_1_3_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff1_3.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_3.reshape((1, model.model.hyperparameters.get("hidden_dim"))), team_1_ave_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff1_ave.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_ave.reshape((1, model.model.hyperparameters.get("hidden_dim")))), 0))
                        com2 = model.transformer_encoder(torch.cat((com2.reshape((7, model.model.hyperparameters.get("hidden_dim"))), team_2_1_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff2_1.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_1.reshape((1, model.model.hyperparameters.get("hidden_dim"))), team_2_2_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff2_2.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_2.reshape((1, model.model.hyperparameters.get("hidden_dim"))), team_2_3_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff2_3.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_3.reshape((1, model.model.hyperparameters.get("hidden_dim"))), team_2_ave_inputs.reshape((1, model.model.hyperparameters.get("hidden_dim"))), diff2_ave.reshape((1, model.model.hyperparameters.get("hidden_dim"))), mul_ave.reshape((1, model.model.hyperparameters.get("hidden_dim")))), 0))
                        com = torch.flatten(torch.cat((com1, com2), 0))
                    
                r_y = model.regressor1(com)
                r_y = model.regressor2(r_y)
                r_y = model.regressor3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y):
                    p.append(r_y)
                    y.append(t[j][2])
                    if output_attention in ["self", "V1", "V2", "None", "encoder"]:
                        if r_y > 0:
                            o.append(t[j][73])
                        else:
                            o.append(t[j][74])
                    elif output_attention in ["V2_reverse", "co", "encoder_all"]:
                        if r_y > 0:
                            o.append(t[j][193])
                        else:
                            o.append(t[j][194])
                    if loss_type == "nba_mae":
                        loss += torch.sqrt((r_y - t[j][2]) ** 2)
                    elif loss_type == "nba_rmse":
                        loss += ((r_y - t[j][2]) ** 2)
                    game += 1

    loss = loss / game
    if loss_type == "nba_rmse":
        loss = torch.sqrt(loss)

    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss

    return loss + reg_loss, p, y, o

def nba_loss_funtion_with_regularizer_loss_only_team(inputs, targets, model, loss_type:str="nba_mae", lamda=1.5e-3):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    loss = 0.0
    p, y, o = list(), list(), list()
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 or t[j][1] != 0:
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
                # no enoder
                # com1 = torch.cat((inp[int(t[j][0])], team_1_ave_inputs), 0)
                # com2 = torch.cat((inp[int(t[j][1])], team_2_ave_inputs), 0)
                # enoder
                com1 = model.transformer_encoder(torch.cat((inp[int(t[j][0])], team_1_ave_inputs), 0).reshape((2, model.model.hyperparameters.get("hidden_dim"))))
                com2 = model.transformer_encoder(torch.cat((inp[int(t[j][1])], team_2_ave_inputs), 0).reshape((2, model.model.hyperparameters.get("hidden_dim"))))
                com1 = com1.reshape((2, model.model.hyperparameters.get("hidden_dim")))
                com2 = com2.reshape((2, model.model.hyperparameters.get("hidden_dim")))
                
                com = torch.cat((com1, com2), 0)
                r_y = model.regressor1(com)
                r_y = model.regressor2(r_y)
                r_y = model.regressor3(r_y)
                if False in torch.isnan(r_y):
                    p.append(r_y)
                    y.append(t[j][2])
                    if r_y > 0:
                        o.append(t[j][73])
                    else:
                        o.append(t[j][74])
                    if loss_type == "nba_mae":
                        loss += torch.sqrt((r_y - t[j][2]) ** 2)
                    elif loss_type == "nba_rmse":
                        loss += ((r_y - t[j][2]) ** 2)
                    game += 1

    loss = loss / game
    if loss_type == "nba_rmse":
        loss = torch.sqrt(loss)

    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss

    return loss + reg_loss, p, y, o

def nba_loss_funtion_with_regularizer_loss_T2T(inputs, targets, model, loss_type:str="nba_mae", lamda=1.5e-3):
    leng = inputs.shape[0]
    game = 0.0
    loss = 0.0
    p, y, o = list(), list(), list()
    for i in range(leng):
        r_y = model.regressor1(inputs[i])
        r_y = model.regressor2(r_y)
        if False in torch.isnan(r_y):
            p.append(r_y)
            y.append(targets[i][0])
            if r_y > 0:
                o.append(targets[i][1])
            else:
                o.append(targets[i][2])
            if loss_type == "nba_mae":
                loss += torch.sqrt((r_y - targets[i][0]) ** 2)
            elif loss_type == "nba_rmse":
                loss += ((r_y - targets[i][0]) ** 2)
            game += 1

    loss = loss / game
    if loss_type == "nba_rmse":
        loss = torch.sqrt(loss)

    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss

    return loss + reg_loss, p, y, o

def nba_loss_funtion_with_regularizer_loss_with_seq(inputs, targets, model, loss_type:str="nba_mae", output_attention:str="None", lamda=1.5e-3):
    seq_dim, leng, _1, _2 = inputs.shape
    assert leng == targets.shape[0]
    game = 0.0
    loss = 0.0
    p, y, o = list(), list(), list()
    for i in range(leng):
        inp = inputs[:, i, :, :]
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
                team_1 = inp[:, int(t[j][0]), :]
                team_2 = inp[:, int(t[j][1]), :]
                st11 = inp[:, team_1_list_st[0], :]
                st12 = inp[:, team_1_list_st[1], :]
                st13 = inp[:, team_1_list_st[2], :]
                st14 = inp[:, team_1_list_st[3], :]
                st15 = inp[:, team_1_list_st[4], :]
                st21 = inp[:, team_2_list_st[0], :]
                st22 = inp[:, team_2_list_st[1], :]
                st23 = inp[:, team_2_list_st[2], :]
                st24 = inp[:, team_2_list_st[3], :]
                st25 = inp[:, team_2_list_st[4], :]
                team_1_mean = torch.mean(inp[:, team_1_list_b, :], dim=1)
                team_2_mean = torch.mean(inp[:, team_2_list_b, :], dim=1)
                com1 = torch.cat((team_1, st11, st12, st13, st14, st15, team_1_mean), 1)
                com2 = torch.cat((team_2, st21, st22, st23, st24, st25, team_2_mean), 1)
                com1 = model.seq_attention(com1)[-1]
                com2 = model.seq_attention(com2)[-1]
                # delete
                if output_attention in ["self", "V1", "V2", "None"]:
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
                    # com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])], st11, st12, st13, st14, st15, st21, st22, st23, st24, st25, team_1_mean, team_2_mean, team_1_ave_inputs, team_2_ave_inputs), 0)
                    if output_attention == "self":
                        com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean, team_1_ave_inputs), 0))
                        com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean, team_2_ave_inputs), 0))
                        com = torch.flatten(torch.cat((com1, com2), 0))
                    if output_attention == "V1":
                        # output attention V1
                        com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), team_1_ave_inputs)
                        com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), team_2_ave_inputs)
                        com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                    elif output_attention == "V2":
                        # output attention V2
                        diff1 = torch.sub(team_1_ave_inputs, team_2_ave_inputs)
                        diff2 = torch.sub(team_2_ave_inputs, team_1_ave_inputs)
                        mul = torch.mul(team_1_ave_inputs, team_2_ave_inputs)
                        com1 = model.attentionLayer(torch.cat((inp[int(t[j][0])], st11, st12, st13, st14, st15, team_1_mean), 0), torch.cat((team_1_ave_inputs, diff1, mul), 0))
                        com2 = model.attentionLayer(torch.cat((inp[int(t[j][1])], st21, st22, st23, st24, st25, team_2_mean), 0), torch.cat((team_2_ave_inputs, diff2, mul), 0))
                        com = torch.cat((com1, com2, team_1_ave_inputs, team_2_ave_inputs), 0)
                    elif output_attention == "None":
                        # no attention
                        com1 = torch.cat((com1, team_1_ave_inputs), 0)
                        com2 = torch.cat((com2, team_2_ave_inputs), 0)
                        com = torch.cat((com1, com2), 0)
                        # com = com.reshape((16, model.model.hyperparameters.get("hidden_dim")))
                        # com = model.transformer_encoder(com)
                        # com = torch.flatten(com)
                elif output_attention in ["V2_reverse", "co"]:
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
                    if output_attention == "V2_reverse":
                        ## output attention reverse
                        com1 = model.attentionLayer(torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0), com1, 12)
                        com2 = model.attentionLayer(torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0), com2, 12)
                        com = torch.cat((com1, com2), 0)
                    elif output_attention == "co":
                        # output co-attention
                        com1, ave1 = model.attentionLayer(com1, torch.cat((team_1_1_inputs, diff1_1, mul_1, team_1_2_inputs, diff1_2, mul_2, team_1_3_inputs, diff1_3, mul_3, team_1_ave_inputs, diff1_ave, mul_ave), 0))
                        com2, ave2 = model.attentionLayer(com2, torch.cat((team_2_1_inputs, diff2_1, mul_1, team_2_2_inputs, diff2_2, mul_2, team_2_3_inputs, diff2_3, mul_3, team_2_ave_inputs, diff2_ave, mul_ave), 0))
                        com = torch.flatten(torch.cat((com1, com2, ave1, ave2), 0))
                    
                r_y = model.regressor1(com)
                r_y = model.regressor2(r_y)
                r_y = model.regressor3(r_y)
                r_y = model.regressor4(r_y)
                if False in torch.isnan(r_y):
                    p.append(r_y)
                    y.append(t[j][2])
                    if output_attention in ["self", "V1", "V2", "None"]:
                        if r_y > 0:
                            o.append(t[j][73])
                        else:
                            o.append(t[j][74])
                    elif output_attention in ["V2_reverse", "co"]:
                        if r_y > 0:
                            o.append(t[j][193])
                        else:
                            o.append(t[j][194])
                    if loss_type == "nba_mae":
                        loss += torch.sqrt((r_y - t[j][2]) ** 2)
                    elif loss_type == "nba_rmse":
                        loss += ((r_y - t[j][2]) ** 2)
                    game += 1

    loss = loss / game
    if loss_type == "nba_rmse":
        loss = torch.sqrt(loss)

    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss

    return loss + reg_loss, p, y, o