import os
import torch

from torch import nn
from tqdm.auto import tqdm
from model.unet import UNet
import torchvision


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
        self.register_buffer('reciprocal_sqrt_alpha_cumprod', torch.sqrt(1. / alpha_cumprod))
        self.register_buffer("remove_noise_coeff", torch.sqrt(1 - alpha_cumprod)/torch.sqrt(alpha_cumprod))
        self.register_buffer("sigma", torch.sqrt(beta))

    @torch.no_grad()
    def sample(self, x_t, t: int, sample_num=1, labels=None,eta=0.0):
        times = torch.full((sample_num,), t, device=self.device, dtype=torch.long)  # times
        #########################################################################prediction
        noise = self.model(x_t, times, labels)
        #########################################################################predict_start_from_noise
        # x_0
        reciprocal_sqrt_alpha_cumprod = self.reciprocal_sqrt_alpha_cumprod.gather(-1, times).reshape(sample_num, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.gather(-1, times).reshape(sample_num, 1, 1, 1)
        x_0= reciprocal_sqrt_alpha_cumprod * (x_t - sqrt_one_minus_alpha_cumprod * noise)
        if t == 0: return x_0  # 返回
        # 第一项系数
        sqrt_alpha_cumprod_prev = self.sqrt_alpha_cumprod.gather(-1, (times - 1).clamp(0)).reshape(sample_num, 1, 1, 1)
        # 第二项系数
        alpha_cumprod_prev = self.alpha_cumprod.gather(-1, (times - 1).clamp(0)).reshape(sample_num, 1, 1, 1)
        #
        alpha_cumprod = self.alpha_cumprod.gather(-1, times).reshape(sample_num, 1, 1, 1)
        sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
        noise_term = sigma * torch.randn_like(x_t)  # 额外噪声

        # 计算 x_t-1
        x_t_prev = sqrt_alpha_cumprod_prev * x_0 + torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise + noise_term
        return x_t_prev

        sigma = self.sigma.gather(-1, times).reshape(sample_num, 1, 1, 1)
        x_pre = x_pre + sigma * (torch.randn_like(x_pre))
        return x_pre

    @torch.no_grad()
    def sample_loop(self, sample_num=16, labels=None, ddim_steps=200, eta=0.0):
        shape = (sample_num, self.channels, self.image_size, self.image_size)
        img = torch.randn(size=shape, device=self.device)  # 随机噪声 均值为0 方差为1 但是范围不定
        times = torch.linspace(self.timesteps - 1, 0, ddim_steps, device=self.device).long()  # [200]
        for t in tqdm(times, desc='DDIM sampling', total=ddim_steps):
            img = self.sample(img, t.item(), sample_num=sample_num, labels=labels, eta=eta)
            img = torch.clamp_(img, min=-1, max=1)

        img = (img + 1) * 0.5  # 归一化到 [0,1]
        return img


if __name__ == '__main__':
    train_steps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet(T=train_steps, ch=16, ch_mult=[1, 4, 8], attn=[2], num_res_blocks=2, dropout=0.15).to(device)
    diffusion = GaussianDiffusion(
        unet,
        timesteps=train_steps,
        batch=256,
        device=device,
    )
    # diffusion.load_state_dict(torch.load("./../model.pth"))
    diffusion = diffusion.to(device)
    diffusion.eval()
    images = diffusion.sample_loop(10, torch.arange(10).to(device))
    # os.makedirs(f"../results/ddim/", exist_ok=True)
    # for i in range(10):
    #     torchvision.utils.save_image(images[i], f'./results/ddim/{i}.jpg')
