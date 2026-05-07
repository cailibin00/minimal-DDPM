"""
GaussianDiffusion implementation
"""
from torch.nn import Module

class GaussianDiffusion(Module):
    """
    1. 根据mode，用数据集来构建监督对象，供模型计算损失\n
    2. 接受model，用于model前向传播
    3. 利用前向传播过程，来实现噪声图像到真实图像的推理过程

    :param model: 用于预测噪声的模型，输入是噪声图像和时间步长，输出是预测的噪声
    :param mode: 预测模式，可选"pred_noisy"
    """
    def __init__(self ,
                 model:Module , mode = "pred_noisy" ):
        super().__init__()
