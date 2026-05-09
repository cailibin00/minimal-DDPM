"""
这里给出t的处理器
"""
import torch , math
import torch.nn as nn
from torch.nn import Module,ModuleList
from einops import rearrange

class TMLP(Module):
    """
    全连接层，给TFilm和Tpreprocessor提供t作为全连接组件
    """
    def __init__(self, time_emb_dim:int=16 , outdim:int = 64 , loops:int= 2):
        super().__init__()
        self.net = ModuleList()
        self.net.append(
                nn.Linear(time_emb_dim , outdim)
            )
        self.net.append(nn.GELU())
        for _ in range(loops-1):
            self.net.append(
                nn.Linear(outdim , outdim)
            )
            self.net.append(nn.GELU())
        

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    

class PosEmb(Module):
    """
    t的位置编码\n
    :input: t : [b] 代表每个样本的时间步长
    :output: [b , time_emb_dim] 代表每个样本的时间步长编码
    """
    def __init__(self, time_emb_dim:int = 16 , type:str = "sin"):
        super().__init__()
        if type not in ["sin" , "fourier"]:
            raise ValueError("type must be one of 'sin' or 'fourier'")
        
        if time_emb_dim % 2 != 0:
            raise ValueError("time_emb_dim must be even")
        
        self.half_dim = time_emb_dim // 2
        self.emtype = type
        self.weights = None
        if type == "fourier":
            self.weights = torch.nn.Parameters(
                torch.randn(self.half_dim) , requires_grad = True
            )
    
    def forward(self , x:torch.Tensor ):
        """
        x : shape [batch , ]
        ->
        output : shape [batch , emb_dim / emb_dim + 1] 
        
        """
        device = x.device
        if self.emtype == "sin":
            emb = math.log(10000) / (self.half_dim)
            emb = torch.exp( torch.arange(self.half_dim) * -emb).to(device)
            out = x[:,None] * emb[None , :] # (batch , half dim)
            out = torch.cat([torch.sin(out) , torch.cos(out)] , dim = -1).to(device)
            return out
        elif self.emtype == "fourier":
            out = x[:,None] * self.weights[None,:] * math.pi * 2
            out = torch.cat([x ,torch.sin(out) , torch.cos(out)] , dim=-1).to(device)
        return 


class TFilm(Module):
    """
    t在各个block之中的调制信号
    """
    def __init__(self, time_emb_dim:int , outdim:int):
        super().__init__()
        self.scale = nn.Linear(time_emb_dim , outdim)
        self.shift = nn.Linear(time_emb_dim , outdim)

    def forward(self, t):
        scale = self.scale(t)
        shift = self.shift(t)
        return (scale , shift)


