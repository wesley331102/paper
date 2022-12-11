import torch


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
    return s / num

def get_accuracy(inputs, targets, model):
    assert inputs.shape[0] == targets.shape[0]
    leng = inputs.shape[0]
    game = 0.0
    right = 0.0
    for i in range(leng):
        inp = inputs[i]
        t = torch.reshape(targets[i], (targets[i].shape[1], targets[i].shape[2]))
        for j in range(t.shape[0]):
            if t[j][0] != 0 and t[j][1] != [0]:
                com = torch.cat((inp[int(t[j][0])], inp[int(t[j][1])]), 0)
                real_y = model.regressor(com)
                if t[j][2]*real_y > 0:
                    right += 1
                game += 1

    right = right / game
    return right