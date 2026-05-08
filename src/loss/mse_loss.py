import torch.nn as nn

def mse_loss(pred , target):
    """
    计算pred和target之间的均方误差\n
    pred和target的shape应该是一样的\n
    这个函数是用来计算损失的\n
    """
    return nn.MSELoss()(pred , target)