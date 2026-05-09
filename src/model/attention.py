"""
实现线性注意力和正常注意力
"""

from model import RMSNorm
from torch.nn import Module
import torch.nn as nn
from torch import einsum
import torch
from einops import rearrange , repeat


class Attention(Module):
    """
    注意力实现
    """
    def __init__(self ,dropout:float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    
    def forward(self,q,k,v):
        dim = q.shape[-1]
        attn = einsum(f'b h i d, b h j d -> b h i j',q,k ) / ( dim**0.5)

        attn = attn.softmax(dim=-1)
        out = einsum(f'b h i j, b h j d -> b h i d' , attn , v)
        return out



class MultiHeadAttention(Module):
    """
    多头注意力
    """
    def __init__(self , 
                dim :int ,
                heads:int = 4, 
                head_dim:int = 32,
                num_men_kv:int = 4 ,
                dropout :float = 0.0):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(dim = dim)
        self.q_pro = nn.Conv2d(dim , heads*head_dim , 1 , bias= False)
        self.k_pro = nn.Conv2d(dim , heads*head_dim , 1 , bias= False)
        self.v_pro = nn.Conv2d(dim , heads*head_dim , 1 , bias= False)
        self.men_kv = nn.Parameter(torch.randn(2 , heads , num_men_kv , head_dim ) , requires_grad=True )
        self.attention = Attention(dropout=dropout)

        self.out_pro = nn.Conv2d(heads*head_dim , dim , 1)

    def forward(self , x:torch.Tensor):
        b , c , h , w = x.shape
        device = x.device
        x = self.norm(x)

        # shape : [batch , C , H , W]
        q = rearrange(self.q_pro(x) ,  'b (head c) h w -> b head (h w) c' , head = self.heads)
        k = rearrange(self.k_pro(x) ,  'b (head c) h w -> b head (h w) c' , head = self.heads)
        v = rearrange(self.v_pro(x) ,  'b (head c) h w -> b head (h w) c' , head = self.heads)

        mk = repeat(self.men_kv[0] , "h len dim -> b h len dim" , b=b)
        mv = repeat(self.men_kv[1] , "h len dim -> b h len dim" , b=b)
        k = torch.cat([ k , mk] , dim = -2).to(device) 
        v = torch.cat([ v , mv] , dim = -2).to(device)

        # shape : [batch , heads , h*w , c]
        out = self.attention(q , k , v)
        out = rearrange(out , 'b head (h w) c -> b (head c) h w' , h = h , w = w)
        out = self.out_pro(out)

        return out + x



class LinearAttention(Module):
    """
    线性注意力：节省显存
    """
    pass



if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.randn( (4 , 3 , 24 , 24)).to(device)
    att = MultiHeadAttention(dim=3 , heads=4 , head_dim=3).to(device)
    out = att(x)
    print(out.shape)
