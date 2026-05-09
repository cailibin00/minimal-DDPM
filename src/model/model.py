"""
储存一些基本的模型组件，是组成后面unet的各个零件，以及为了方便复现
"""

# 要支持is_conditional这个参数
# 支持的模型，有iamge , condition , timesteps三种输入 
# 否则就是 image , timesteps两种输入



"""
Unet:
先用卷积把输入映射到特征空间
在下采样路径不断提取更抽象的多尺度特征
在瓶颈位置做最强的全局交互
在上采样路径逐层恢复分辨率，并通过 skip connection 融合浅层细节
最后输出与输入通道对齐的结果
"""

"""
首先是unet的核心，分辨率压缩和恢复的两个核心组件
"""
# 上采样
from torch.nn import Module
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from t_processor import TMLP , PosEmb , TFilm

def default(value , dft):
    """
    缺省设置
    """
    return dft if value is None else value

def exists(value):
    """
    判断是否存在
    """
    return value is not None

def Upsample(indim:int , outdim:int=None):
    """
    采用的上采样策略是直接复制
    conv3*3 用于局部重建
    """
    outdim = default(outdim , indim)
    return nn.Sequential(
        nn.Upsample(scale_factor=2 , mode='nearest'),
        nn.Conv2d(indim , outdim , kernel_size=3 , padding=1)
    )


def Downsample(indim:int , outdim:int=None):
    """
    采用的下采样策略是将相邻像素压缩到同一个通道，然后卷积处理
    conv3*3 用于局部重建
    """
    outdim = default(outdim , indim)
    return nn.Sequential(
        Rearrange('b c (h0 scale_h) (w0 scale_w) -> b (c scale_h scale_w) h0 w0' , scale_h = 2 , scale_w=2),
        nn.Conv2d(indim*4 , outdim , kernel_size=3 , padding=1)
        )
    

class RMSNorm(Module):
    """
    对于每一个通道做归一化
    """
    def __init__(self , dim:int , eps:float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1 , dim , 1 , 1))
        self.shift = nn.Parameter(torch.zeros(1 , dim , 1 , 1))

    def forward(self , x:torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected x to have shape [batch, channel, height, width], but got {tuple(x.shape)}")

        rms = torch.sqrt(torch.mean(x ** 2 , dim=1 , keepdim=True) + self.eps)
        return (x / rms) * self.scale + self.shift


class ResnetBlock(Module):
    """
    3x3 Conv
    RMSNorm
    SiLU
    Dropout
    """
    def __init__(self , indim:int , outdim:int , dropout:float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(indim , outdim , kernel_size = 3 , padding = 1)
        self.norm = RMSNorm(outdim)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self , x:torch.Tensor , film:tuple= None):
        x = self.conv(x)
        x = self.norm(x)
        if exists(film):
            scale , shift = film
            if scale.dim() != 4 :
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * (scale + 1) + shift
        x = self.silu(x)
        return self.dropout(x)

# TODO:对于归一化的位置有疑问
class Resnet(Module):
    """
    代码参考里面的实现
    1. mlp ： 对于t的block处理
    2. 单个block注入时间信息，第一个
    3. 
    
    """
    def __init__(self ,
        indim:int , outdim:int , 
        time_emb_dim:int,
        dropout:float = 0.0):
        super().__init__()
        self.t_mlp = TMLP(time_emb_dim , outdim=outdim)
        self.film = TFilm( outdim , outdim)
        self.block1 = ResnetBlock(indim = indim , outdim= outdim , dropout=dropout)
        self.block2 = ResnetBlock(indim = outdim , outdim= outdim )
        self.out_conv = nn.Conv2d(indim , outdim , kernel_size = 3 ,padding =1)

    def forward(self , x:torch.Tensor , time_emb:torch.Tensor):
        time_emb = self.t_mlp(time_emb)
        film = self.film(time_emb)
        h = self.block1(x , film)
        h = self.block2(h)
        return h + self.out_conv(x)










