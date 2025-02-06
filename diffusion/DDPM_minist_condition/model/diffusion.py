import sys
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)  # 0.001 0.2 100


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=100, batch=16, device="cpu"):
        super().__init__()
        self.model = model
        self.channels = 1
        self.batch = batch
        self.image_size = 28
        self.device = device
        self.timesteps = timesteps
        self.criterion = torch.nn.MSELoss().to(device)
        self._register_buffer(timesteps)

    def _register_buffer(self, timesteps) -> None:
        ######################################################################################################基础参数
        beta = linear_beta_schedule(timesteps)  # beta [0.001->0.2] 100 step
        alpha = 1. - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1. - alpha_cumprod))
        ######################################################################
        self.register_buffer('reciprocal_sqrt_alpha', torch.sqrt(1. / alpha))
        self.register_buffer("remove_noise_coeff", beta / torch.sqrt(1 - alpha_cumprod))
        self.register_buffer("sigma", torch.sqrt(beta))

    @torch.no_grad()
    def sample(self, x_t, t: int, sample_num=1):
        times = torch.full((sample_num,), t, device=self.device, dtype=torch.long)  # times
        #########################################################################prediction
        noise = self.model(x_t, times)
        #########################################################################predict_start_from_noise
        remove_noise_coeff = self.remove_noise_coeff.gather(-1, times).reshape(sample_num, 1, 1, 1)
        reciprocal_sqrt_alpha = self.reciprocal_sqrt_alpha.gather(-1, times).reshape(sample_num, 1, 1, 1)
        _x = reciprocal_sqrt_alpha * (x_t - remove_noise_coeff * noise)

        sigma = self.sigma.gather(-1, times).reshape(sample_num, 1, 1, 1)
        if t > 0: _x = _x + sigma * (torch.randn_like(_x).clamp_(min=-1, max=1))
        return _x

    @torch.no_grad()
    def sample_loop(self, sample_num=16):
        shape = (sample_num, self.channels, self.image_size, self.image_size)
        img = torch.randn(size=shape, device=self.device).clamp_(-1, 1)  # 随机噪声 均值为0 方差为1 但是范围不定
        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling', total=self.timesteps):
            img = self.sample(img, t, sample_num=sample_num)
        img = torch.clamp_(img, min=-1, max=1)
        img = (img + 1) * 0.5
        return img

    def xt_sample(self, x_0, time_step, noise):
        # 按照时间步抽取对应系数
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.gather(-1, time_step).view(self.batch, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.gather(-1, time_step).view(self.batch, 1, 1, 1)
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

    def forward(self, x_0,labels):
        self.batch = x_0.shape[0]
        # 随机生成不同img的时间步
        time_step = torch.randint(0, self.timesteps, (self.batch,), device=self.device).long()
        noise = torch.randn_like(x_0)

        x_t = self.xt_sample(x_0=x_0, time_step=time_step, noise=noise)  # [b c h w]
        t_noise = self.model(x_t, time_step)  # xt = x+noise ->[3, 3, 64, 64] 预测的是噪声
        loss = self.criterion(t_noise, noise)
        return loss
