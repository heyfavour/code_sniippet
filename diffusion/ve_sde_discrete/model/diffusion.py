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


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)
        self.dt = -1. / self.sde.N

    def update_fn(self, model, x, t, y):
        z = torch.randn_like(x)
        drift, diffusion = self.reverse(model, x, t, y)
        x_mean = x + drift * self.dt
        x = x_mean + diffusion * np.sqrt(-self.dt) * z
        return x, x_mean

    def reverse(self, model, x_t, t, y):
        # 跟predictor update function有关系
        score = self.sde.score_fn(model, x_t, t, y)
        drift, diffusion = self.sde.sde(x_t, t)
        drift = drift - diffusion ** 2 * score
        return drift, diffusion


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, model, x_t, t, y):
        f, G = self.reverse(model, x_t, t, y)
        z = torch.randn_like(x_t)
        x_mean = x_t - f
        x = x_mean + G * z
        return x, x_mean

    def reverse(self, model, x_t, t, y):
        f, G = self.sde.discretize(x_t, t)
        rev_f = f - G ** 2 * self.sde.score_fn(model, x_t, t, y)
        rev_G = G
        return rev_f, rev_G


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps


class LangevinCorrector(Corrector):
    def __init__(self, sde, snr, n_steps):
        super().__init__(sde, snr, n_steps)

    def update_fn(self, model, x_t, t, y):
        for i in range(self.n_steps):
            grad = self.sde.score_fn(model, x_t, t, y)
            noise = torch.randn_like(x_t)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (((self.snr * noise_norm / grad_norm) ** 2 * 2) * torch.ones_like(t))[:, None, None, None]
            x_mean = x_t + step_size * grad
            x_t = x_mean + torch.sqrt(step_size * 2) * noise

        return x_t, x_mean


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
        self.smld_sigma = torch.flip(self.sde.sigmas, dims=(0,))
        # self.corrector = LangevinCorrector(self.sde,0.3,2)
        # self.predictor = ReverseDiffusionPredictor(self.sde)
        self.predictor = EulerMaruyamaPredictor(self.sde)

    def forward(self, x_0, label):
        self.batch = x_0.shape[0]
        # 随机生成不同img的时间步
        timestep = torch.randint(0, self.sde.N, (self.batch,), device=self.device)  # [0 1 2 3 ... 998 999]
        sigma = self.smld_sigma[timestep]  # timestep越大 噪声越大
        noise = torch.randn_like(x_0) * sigma[:, None, None, None]  # 均值为0 反差为sigma
        x_t = x_0 + noise
        score = self.model(x_t, t=timestep, y=label)

        target = - noise / (sigma ** 2)[:, None, None, None]  # z/std^2

        # loss = torch.mean(torch.mean(torch.square(score - target).reshape(self.batch, -1), dim=-1) * sigma ** 2)
        loss = self.criterion(score, target)
        return loss

    @torch.no_grad()
    def sample(self, sample_num=16, labels=None):
        shape = (sample_num, self.channels, self.image_size, self.image_size)
        x = self.sde.prior_sampling(shape).to(self.device)
        timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)
        # timestep 越小噪声越小
        for i in range(self.sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            x, x_mean = self.predictor.update_fn(self.model, x, vec_t, y=labels)

        img = torch.clamp_(x_mean, min=-1, max=1)
        img = (img + 1) * 0.5
        return img
