import torch


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
                real_y = model.regressor(com)
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
                y.append(t[j][2])
    return p, y