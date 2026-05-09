"""
GaussianDiffusion implementation
"""
from pathlib import Path
import sys
import torch
from loss import mse_loss
from torch.nn import Module
from torchvision.utils import save_image

def normalize(x):
    x= x*2 -1
    return x

def denormalize(x):
    x = (x + 1) / 2
    return x

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

# TODO:方差可以给模型一起来预测
class GaussianDiffusion(Module):
    """
    1. 根据mode，用数据集来构建监督对象，供模型计算损失\n
    2. 接受model，用于model前向传播
    3. 利用前向传播过程，来实现噪声图像到真实图像的推理过程

    :param model: 用于预测噪声的模型，输入是噪声图像和时间步长，输出是预测的噪声
    :param image_size: 图像的大小
    :param timesteps: 扩散过程的时间步长数量
    :param mode: 预测模式，可选"pred_noisy"
    :param is_conditional: 是否是条件生成模型，如果是条件生成模型，则需要在前向传播过程中输入条件信息(目前不支持)
    :param forward_steps: 推理过程中，每次推理的步数 （由于后期用来做实验）

    ## Functions\n
    :forward: 计算损失的函数
    :model_out: 根据mode，用model来计算out
    :inference_batch: 根据mode，使用模型预测得到的噪声买得到真实图像
    :inference_single_image: 根据mode，使用模型预测得到的噪声买得到真实图像

    """
    def __init__(self ,
                 model:Module , image_size:int ,
                 timesteps:int = 1000,
                 mode:str = "pred_noisy",
                 forward_steps:int = 1000,
                 is_conditional:bool = False,
                 device:str = "cuda"):
        super().__init__()
        self.device = device
        self.mode = mode
        self.model = model.to(device)
        self.is_conditional = is_conditional
        self.image_size = image_size

        if self.model.is_conditional != self.is_conditional:
            raise ValueError(f"Model's is_conditional attribute ({self.model.is_conditional}) does not match the GaussianDiffusion's is_conditional attribute ({self.is_conditional}). Please ensure they are consistent.")
        
        self.timesteps = timesteps
        self.f_steps = forward_steps
        # 要来注册alpha_bar , beta_bar , alpha , beta
        # 公式beta = 1- alpha
        # 公式alpha_bar = alpha_1 * alpha_2 * ... * alpha_t
        # alpha_bar= 1- beta_bar
        betas = linear_beta_schedule(timesteps).to(device)
        betas = betas.view(-1 , 1 , 1 , 1) # [timesteps , 1 , 1 , 1]
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas , dim=0).to(device) # [timesteps , 1 , 1 , 1]
        alpha_bars_prev = torch.cat([torch.ones((1 , 1 , 1 , 1) , dtype=alphas.dtype ,device=device) , alpha_bars[:-1]] , dim=0)
        beta_bars = 1 - alpha_bars
        posterior_variance = betas * (1. - alpha_bars_prev) / (1. - alpha_bars)

        register_buffer = lambda name , value : self.register_buffer(name , value.to(torch.float32))
        register_buffer("betas" , betas)
        register_buffer("alphas" , alphas)
        register_buffer("alpha_bars" , alpha_bars)
        register_buffer("alpha_bars_prev" , alpha_bars_prev)
        register_buffer("beta_bars" , beta_bars)
        register_buffer("sqrt_alpha_bars" , torch.sqrt(alpha_bars ).to(device))
        register_buffer("sqrt_beta_bars" , torch.sqrt(beta_bars ).to(device))
        register_buffer("posterior_variance" , posterior_variance)

    # DONE
    def _is_true_shape(self , images ):
        b , c , h , w = images.shape
        assert h == self.image_size and w == self.image_size , f"Expected image size: {self.image_size}x{self.image_size}, but got {h}x{w}"

    # 根据我们的办法，可视化单张/batch图片的加噪过程，记录的图片是原始图片，一点噪声的图片，...，完全噪声的图片
    @torch.inference_mode()
    def _vis_forward_process_images(
        self ,
        images:torch.Tensor,
        num_save:int = 2 ,
        save_dir:Path | str | None = None,
        num_frames:int = 20,
        prefix:str = "",
        input_is_normalized:bool = False
    ):
        # 检查维度，batch处理
        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise ValueError(f"Expected images to have 3 or 4 dimensions, but got {images.dim()} dimensions.")
        self._is_true_shape(images)

        # 参数处理
        if num_save <= 0:
            raise ValueError(f"Expected num_save > 0, but got {num_save}.")
        if num_frames <= 0:
            raise ValueError(f"Expected num_frames > 0, but got {num_frames}.")

        # 路径处理
        if save_dir is None:
            save_dir = Path(__file__).resolve().parent.parent / "results" / "forward_process_images"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 取前面几张图片
        images = images[: min(num_save, images.shape[0])].to(self.device).float()
        if input_is_normalized:
            diffusion_images = images.clamp(-1.0, 1.0)
            images_to_save = denormalize(diffusion_images).clamp(0.0, 1.0)
        else:
            diffusion_images = normalize(images)
            # 截断
            images_to_save = images.clamp(0.0, 1.0)

        # t采样
        sampled_ts = torch.linspace(
            0 ,
            self.timesteps - 1 ,
            steps=min(num_frames , self.timesteps) ,
            device=self.device
        ).round().long().unique(sorted=True)

        saved_paths = []
        original_path = save_dir / f"{prefix}_original.png"
        save_image(
            images_to_save.cpu() ,
            original_path ,
            nrow=min(diffusion_images.shape[0] , 8)
        )
        saved_paths.append(str(original_path))

        for i in sampled_ts.tolist():
            noisy = torch.randn_like(diffusion_images)
            xt = diffusion_images * self.sqrt_alpha_bars[i] + noisy * self.sqrt_beta_bars[i]
            save_path = save_dir / f"{prefix}_t_{i:04d}.png"
            save_image(
                denormalize(xt).clamp(0.0 , 1.0).cpu() ,
                save_path ,
                nrow=min(diffusion_images.shape[0] , 8)
            )
            saved_paths.append(str(save_path))

        return saved_paths

    @torch.inference_mode()
    def vis_forward_process_images(
        self ,
        images:torch.Tensor,
        num_save:int = 2 ,
        save_dir:Path | str | None = None,
        num_frames:int = 20,
        prefix:str = "forward_process",
        input_is_normalized:bool = False
    ):
        return self._vis_forward_process_images(
            images=images,
            num_save=num_save,
            save_dir=save_dir,
            num_frames=num_frames,
            prefix=prefix,
            input_is_normalized=input_is_normalized
        )

    # DONE
    def _get_supervision(self , images , ts):
        """
        根据mode，来计算相应的监督对象\n
        支持的mode:\n
        1. pred_noisy: 预测噪声
        2. pred_x0: 预测原始图像
        3. pred_v: 预测v，v是噪声图像和原始图像的线性组合

        :return:xt , t , supervision_target

        """
        # 首先t已经获得
        # 接着计算noisy，再根据噪声来预测各个结果
        noisy = torch.randn_like(images)  # 噪声就是高斯分布N(0,1)

        # 根据alpha_bar还有beta_bar来计算噪声图像
        # images: [batch , c , h , w]
        # noisy: [batch , c , h , w]
        # alpha_bar: [timesteps ,] -> [timesteps , 1 , 1 , 1]
        xt = images * self.sqrt_alpha_bars[ts] + noisy * self.sqrt_beta_bars[ts]
        mode = self.mode
        if mode == "pred_noisy":
            supervision_target = noisy
        elif mode == "pred_x0":
            supervision_target = images
        elif mode == "pred_v":
            pass
        
        return xt , ts , supervision_target
    
    # DONE
    def _compute_loss(self , preds , targets ):
        """
        传入模型预测的结果还有监督对象，直接计算损失
        """
        return mse_loss(preds , targets)

    # DONE
    def model_out(self , xt:torch.Tensor , t:torch.Tensor , condition = None ):
        """
        模型推理的时候的前向传播，肯定是根据mode来进行的\n

        :return: 推理结果，是三个mode之一
        """
        return self.model(xt , t , condition)
    
    # TODO：下面这两个函数要实现modelout : variance
    @torch.inference_mode()
    def _forward_pred_noisy_loops(self , xt , t:int , condition=None , add_noisy:bool=True , return_process:bool = False):
        """
        xt:shape [batch , c , h , w]

        根据噪声，一步一步向前推进，从x{t} -> x_{t-1}\n
        formula: \n
        1. x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t , t))
        2. x_{t-1} = x_{t-1} + sigma_t * z (如果add_noisy为True) (z = N() , sigma_t^2 = posterior_variance_t )
        """
        xt = xt
        results = []

        # 计算推理和训练的差距时间，用来正确采样
        delta_t = self.timesteps // t

        for i in range(t - 1 , -1 , -1):
            tt = int(i * delta_t)
            t_tensor = torch.full((xt.shape[0],) , tt , device=xt.device , dtype=torch.long)
            out = self.model_out(xt , t_tensor , condition)
            u = (xt - (self.betas[tt] / self.sqrt_beta_bars[tt]) * out) / torch.sqrt(self.alphas[tt])
            if add_noisy and tt > 0:
                noisy = torch.randn_like(xt)
                u = u + torch.sqrt(self.posterior_variance[tt]) * noisy
            if return_process: results.append(u)    
            xt = u
        
        return results if return_process else xt

    # TODO:实现模型预测的x0的可视化
    @torch.inference_mode()
    def _forward_pred_x0_loops(self, xt:torch.Tensor , t:int , condition=None , add_noisy:bool=True , return_process:bool = False):
        """
        xt:shape [batch , c , h , w]

        根据噪声，一步一步向前推进，从x{t} -> x_{t-1}\n
        formula: \n
        1. x_{t-1} = \sqrt{alpha_t} * beta_bar_{t-1} / (beta_bar_t) * x_t + \sqrt{alpha_bar} * beta_{t} / beta_bar_{t} * x0
        2. x_{t-1} = x_{t-1} + sigma_t * z (如果add_noisy为True) (z = N() , sigma_t = sqrt(beta_t) )
        """
        xt = xt.to(self.device)
        results = []

        noisy = 0

        # 计算推理和训练的差距时间，用来正确采样
        delta_t = self.timesteps // t

        for i in range(t , 0 , -1):
            if add_noisy :
                noisy = torch.randn_like(xt)
            tt = int(i*delta_t)
            out = self.model_out(xt , torch.ones((xt.shape[0],1))*tt , condition)
            u = torch.sqrt(self.alphas[tt]) * (self.beta_bars[tt-1] / self.beta_bars[tt]) * xt + self.sqrt_alpha_bars[tt] * ((1-self.alphas[tt]) / (1-self.alpha_bars[tt])) * out
            u = u + self.sqrt_beta_bars[tt]*noisy # or self.beta[tt] * noisy
            if return_process: results.append(u)    
            xt = u
        
        return results if return_process else xt

    # DONE
    @torch.inference_mode()
    def inference_batch(self , xt:torch.Tensor , condition=None ,forward_steps:int = 0 ,  mode:str="" , return_process:bool = False ):
        """
        是根据mode，来使用模型预测得到的噪声来得到真实图像\n
        同时，这个函数支持外部调用，支持中间xt的输入，然后自定义t，用于最终的实验

        :param mode: 推理模式，支持的mode:\n
        1. pred_noisy: 预测噪声
        2. pred_x0: 预测原始图像
        3. pred_v: 预测v，v是噪声图像和
        :param return_process: 是否返回推理过程中的所有图像，如果为True，则返回一个列表，包含每个时间步的图像,shape:[batch , t,c,h,w]
        
        :return: output_images :shape[batch , t(1/timesteps) , c , h , w]
        """
        # 需要model ， f_steps , mode , 以及对应的公式
        xt = xt.to(self.device)

        if mode!="" and mode!= self.mode:
            print(f"Warning: The specified mode '{mode}' does not match the GaussianDiffusion's mode '{self.mode}'. \nThe inference will proceed using the specified mode, but please ensure that the model is compatible with this mode for accurate results.")
            mode = mode
        else:
            mode = self.mode
        
        if forward_steps == 0 and self.f_steps != 0:
            forward_steps = self.f_steps
        
        output_images = []
        if mode == "pred_noisy":
            output_images = self._forward_pred_noisy_loops(xt ,forward_steps , condition , return_process=return_process)
        elif mode == "pred_x0":
            output_images = self._forward_pred_x0_loops(xt , forward_steps , condition , return_process=return_process)
        elif mode == "pred_v":
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}. Please specify a valid mode from the following options: pred_noisy, pred_x0, pred_v.")

        return output_images

    @torch.inference_mode()
    def inference_single_image(self , xt:torch.Tensor , condition=None , forward_steps:int = 0 ,  mode:str="" , return_process:bool = False):
        """
        推理单个图像
        """
        return self.inference_batch(xt.view(1,-1,-1,-1) ,condition ,forward_steps,  mode , return_process)


    def forward(self , batch , device:str = "cuda"):
        """
        计算损失:\n
        1. 处理输入：device ， normalization
        2. 得到监督对象
        """
        # shape : [batch , ]
        image = None
        condition = None

        # 输入处理
        if self.is_conditional:
            image , condition = batch
            image = image.to(device)
            image = normalize(image.float())
            condition = condition.to(device).float()
            # 这里需要添加形状校验
            self._is_true_shape(image)
        else:
            # 仅仅支持is_conditional为False
            # 要进行归一化
            image = batch.to(device).float()
            self._is_true_shape(image)
            image = normalize(image)

        # 得到监督目标
        timesteps = torch.randint(0 , self.timesteps , (image.shape[0] ,) , device=device)
        xt , t , targets = self._get_supervision(image , timesteps)
        
        # 先输入给模型
        preds = self.model_out(xt, t, condition)

        # 计算得到损失
        return self._compute_loss(preds , targets)
    


            

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.load_data import load_data
    from data.dataset import Dataset
    from torch.utils.data import DataLoader
    class model(Module):
        def __init__(self):
            super().__init__()
            self.is_conditional = False
        def forward(self , xt , t , condition=None):
            return torch.randn_like(xt)

    d = Dataset(image_size=32 , is_conditional=False , dataset_name="CIFAR10" , split="train")
    g = GaussianDiffusion(model = model() , image_size = 32)
    dataloader = DataLoader(d , batch_size=16 , shuffle=True)
    for batch in dataloader:
        g.vis_forward_process_images(batch, num_save = 1)
        l = g.forward(batch , device="cuda")
        print(l)
        print(l.shape)
        break
    
