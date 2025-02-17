import sys
import torch
from torch import nn
from tqdm.auto import tqdm
import abc
import torch
import numpy as np


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(self, x, t, y):
        pass

    def ve_score_fn(self, model, x_t, t, y):
        score = self.sde.score_fn(model, x_t, t, y)
        drift, diffusion = self.sde.sde(x_t, t)
        drift = drift - diffusion ** 2 * score
        return drift, diffusion


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, model, x, t, y):
        dt = -1. / self.sde.N
        z = torch.randn_like(x)
        drift, diffusion = self.reverse(model, x, t, y)
        x_mean = x + drift * dt
        x = x_mean + diffusion * np.sqrt(-dt) * z
        return x, x_mean




class GaussianDiffusion(nn.Module):
    def __init__(self, model, sde, batch=16, device="cpu", eps=1e-5):
        super().__init__()
        self.model = model
        self.sde = sde
        self.channels = 1
        self.batch = batch
        self.image_size = 28
        self.device = device
        self.eps = eps
        self.criterion = torch.nn.MSELoss().to(device)
        self.predictor = EulerMaruyamaPredictor(self.sde)
        # [0.0100, 0.0150, 0.0200 ... , 4.9900, 4.9950, 5.0000]
        self.smld_sigmas = torch.flip(self.sde.sigmas, dims=(0,))

    def forward(self, x_0, label):
        self.batch = x_0.shape[0]
        # 随机生成不同img的时间步
        timestep = torch.randint(0, self.sde.N, (self.batch,), device=self.device)  # [0 1 2 3 ... 998 999]
        sigma = self.smld_sigmas[timestep]
        # (0,1)*#[0.0100, 0.0150, 0.0200 ... , 4.9900, 4.9950, 5.0000]
        noise = torch.randn_like(x_0) * sigma[:, None, None, None]
        x_t = x_0 + noise
        score = self.model(x_t, t=timestep, y=label)

        target = - noise / (sigma ** 2)[:, None, None, None]  # z/std^2

        loss = torch.mean(torch.mean(torch.square(score - target).reshape(self.batch, -1), dim=-1) * sigma ** 2)
        return loss

    @torch.no_grad()
    def sample(self, sample_num=16, labels=None):
        shape = (sample_num, self.channels, self.image_size, self.image_size)
        x = self.sde.prior_sampling(shape).to(self.device)
        # [0.1 ->5] #[0.0100, 0.0150, 0.0200 ... , 4.9900, 4.9950, 5.0000]
        timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)
        for i in tqdm(range(self.sde.N), desc="sample"):  # 1000
            t = timesteps[i]
            vec_t = torch.ones(sample_num, device=t.device) * t
            x, x_mean = self.predictor.update_fn(self.model, x, t=vec_t, y=labels)

        img = torch.clamp_(x_mean, min=-1, max=1)
        img = (img + 1) * 0.5
        return img
