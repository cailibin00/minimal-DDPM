from torch.nn import Module , ModuleList
from attention import MultiHeadAttention
from model import RMSNorm , Upsample , Downsample , Resnet
from t_processor import PosEmb , TMLP
import torch.nn as nn
import torch


def default(key , value):
    return value if not key else key

def cast(multi , dim):
    res = []
    cur = dim
    for i in multi:
        res.append( (cur  , i * dim))
        cur = i * dim
    return res


# TODO:支持多个attention类型
class UnetDownBlock(Module):
    """
    resnet * 2
    atten *1 
    downs * 1
    """
    def __init__(self , 
                 # 输入的各个维度
                indim :int , 
                outdim:int ,
                time_emb_dim:int ,

                # attention配置
                heads:int = 4, 
                num_men_kv:int = 4,
                head_dim:int = 32,

                dropout:float = 0.0,
                downsample:bool = True):
        super().__init__()
        self.res1 = Resnet(indim , indim,  time_emb_dim , dropout)
        self.res2 = Resnet(indim , indim,  time_emb_dim , dropout)
        self.attn = MultiHeadAttention(indim , heads , head_dim , num_men_kv , dropout)
        self.down = Downsample(indim , outdim) if downsample else nn.Conv2d(indim , outdim , kernel_size = 3 ,padding =1)


    def forward(self , x , t):
        x1 = self.res1(x , t)
        x = self.res2(x1 , t)
        x2 = self.attn(x)
        out = self.down(x2)
        return out , x , x1 

class UnetUpBlock(Module):
    """
    resnet * 2
    atten *1 
    Ups * 1
    """
    def __init__(self , 
                 # 输入的各个维度
                indim :int , 
                outdim:int ,
                time_emb_dim:int ,

                # attention配置
                heads:int = 4, 
                num_men_kv:int = 4,
                head_dim:int = 32,

                dropout:float = 0.0,
                upsample:bool = True):
        super().__init__()
        self.res1 = Resnet(indim , indim-outdim,  time_emb_dim , dropout )
        self.res2 = Resnet(indim , indim-outdim,  time_emb_dim , dropout )
        self.attn = MultiHeadAttention(indim-outdim , heads , head_dim , num_men_kv , dropout)
        self.up = Upsample(indim-outdim , outdim) if upsample else nn.Conv2d(indim-outdim , outdim , kernel_size = 3 ,padding =1)



    def forward(self , x , t , h):
        x = torch.cat((x, h[-1]), dim = 1)
        x = self.res1(x , t)
        x = torch.cat((x, h[-2]), dim = 1)
        x = self.res2(x , t)

        out = self.attn(x)
        out = self.up(out)
        return out



# TODO:支持预测方差，支持多种attention
# TODO:支持x_cond
class Unet(Module):
    """
    0. initial image，先把他变成很多个通道
    1. 先处理时间 PosEmb
    2. 在走整个通路，分为三个部分
    3. 输出头：分为可选(variance)
    
    :param init_dim: 初始输入unet处理的dim
    :param dim_mults: 输入的各个维度的倍数
    :param time_emb_dim: 时间嵌入的维度
    :param emb_type: 时间嵌入的类型
    :param heads: 多头注意力的头数
    :param head_dim: 多头注意力维度
    :param num_men_kv: 多头注意力中kv的个数
    :param dropout: 丢弃概率
    :param learn_variance: 是否学习方差
    :param is_conditional: 是否有条件
    """
    def __init__(self , 
                # U的配置
                init_dim:int , # 初始输入unet处理的dim

                dim_mults:tuple = (1 , 2 , 4 , 8),  # dim的翻倍数
                image_dim:int = 3 ,
                # 时间配置
                time_emb_dim:int = 16,
                emb_type : str = "sin",

                # attention配置
                heads:int = 4, 
                head_dim:int = None,
                num_men_kv:int = 4,
                
                # 这个配置起始配置
                dropout:float = 0.0 ,
                learn_variance:bool = False,

                is_conditional:bool = False,

                ):
        super().__init__()
        # 输入的标签
        self.is_conditional = is_conditional
        self.init_dim = default(init_dim , 64)
        self.time_emb_dim = default(time_emb_dim , init_dim//2)
        self.head_dim = default(head_dim , init_dim // 2)
        self.emb_type = emb_type
        self.learn_variance = learn_variance
        self.n_layers = len(dim_mults)

        self.init_block = nn.Conv2d(image_dim , self.init_dim , kernel_size=7 , padding=3)
        self.time_preprocess = nn.Sequential(
            PosEmb(time_emb_dim = self.time_emb_dim , type = emb_type) , 
            TMLP(time_emb_dim = self.time_emb_dim  if emb_type == "sin" else self.time_emb_dim+1 ,
                 outdim = self.time_emb_dim)
            )
        
        in_out = cast(dim_mults , self.init_dim)
        head_in_out = [self.head_dim * i for i in dim_mults ]
        # 构造下采样路径
        self.downs = ModuleList()
        for (indim , outdim),head_dim in zip(in_out , head_in_out) :
            self.downs.append( 
                UnetDownBlock(
                    indim = indim,
                    outdim = outdim,
                    time_emb_dim = self.time_emb_dim,
                    heads = heads ,
                    head_dim = head_dim,
                    num_men_kv = num_men_kv,
                    dropout = dropout,
                    downsample = True   
                )
            )
        # 最后加一层不需要下采用
        self.downs.append( 
            UnetDownBlock(
                indim = in_out[-1][1],
                outdim = in_out[-1][1],
                time_emb_dim = self.time_emb_dim,
                heads = heads ,
                head_dim = head_in_out[-1],
                num_men_kv = num_men_kv,
                dropout = dropout,
                downsample = False   
            )
        )

        self.Mnet = nn.Sequential(
            Resnet(indim = in_out[-1][1] , outdim = in_out[-1][1] , 
                   time_emb_dim = self.time_emb_dim , dropout = dropout),
            MultiHeadAttention(dim = in_out[-1][1] , heads = heads , head_dim = self.head_dim*dim_mults[-1] , 
                               num_men_kv = num_men_kv , dropout = dropout),
            Resnet(indim = in_out[-1][1] , outdim = in_out[-1][1] , 
                   time_emb_dim = self.time_emb_dim , dropout = dropout),
        )


        # 注意
        # 构造下采样路径
        self.ups = ModuleList()
        self.ups.append( 
            UnetUpBlock(
                indim = in_out[-1][1]*2,
                outdim = in_out[-1][1],
                time_emb_dim = self.time_emb_dim,
                heads = heads ,
                head_dim = head_in_out[-1],
                num_men_kv = num_men_kv,
                dropout = dropout,
                upsample = True   
            )
        )
        for i , ((indim , outdim),head_dim) in enumerate(zip(*map(reversed,(in_out , head_in_out)))) :
            self.ups.append( 
                UnetUpBlock(
                    indim = indim + outdim,
                    outdim = indim,
                    time_emb_dim = self.time_emb_dim,
                    heads = heads ,
                    head_dim = head_dim,
                    num_men_kv = num_men_kv,
                    dropout = dropout,
                    upsample = True   if i != self.n_layers - 1 else False
                )
            )
        
        predoutdim = 3 if not self.learn_variance else 6
        self.out_block = nn.Sequential(
            Resnet(indim = self.init_dim * 2 , outdim = init_dim , 
                   time_emb_dim = self.time_emb_dim , dropout = dropout),
            nn.Conv2d(self.init_dim , predoutdim , 1) 
        )
          
    def forward(self , x:torch.Tensor ,t:torch.Tensor ,condition:torch.Tensor = None ):
        if condition is not None or self.is_conditional:
            raise NotImplementedError("condition not implemented")
        
        # 先处理时间 ， 还有图片
        t_emb = self.time_preprocess(t)
        x = self.init_block(x)
        
        x_clone = x.clone()

        history = []

        for block in self.downs :
            x , x1 , x2 = block(x , t_emb)
            history.append((x1 , x2))
        
        x = self.Mnet[0](x , t_emb)
        x = self.Mnet[1](x)
        x = self.Mnet[2](x , t_emb)

        for i , block in enumerate(self.ups):
            x = block(x , t_emb , history[-i-1])
        
        # shape : [b , c , h , w]
        x = torch.cat([x , x_clone] , dim = 1)
        x = self.out_block[0](x , t_emb)
        x = self.out_block[1](x)

        if self.learn_variance:
            return x[:,:3,:,:] , x[:,3:,:,:]
        else:
            return x , torch.zeros_like(x).to(x.device)


if __name__ == "__main__":
    device = torch.device("cuda")
    model = Unet(init_dim = 12 , dim_mults = (1 , 2 , 4 , 8) , heads = 4 , time_emb_dim=12 ,learn_variance=False ,
                 num_men_kv= 4 , is_conditional= True).to(device)
    x = torch.randn( (2 , 3 , 32 , 32)).to(device)
    t = torch.randint(0 , 1000 , (2 , )).to(device)
    out , _ = model(x , t)
    print(out.shape)



