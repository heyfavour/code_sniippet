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


# reverse_diffusion
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, model, x, t, y):
        f, G = self.sde.reverse(model, x, t, y)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G * z
        return x, x_mean


class Corrector(abc.ABC):
    def __init__(self, sde, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, model,x, t, y):
        dt = -1. / self.sde.N
        z = torch.randn_like(x)
        drift, diffusion = self.sde.reverse(model,x, t, y)
        x_mean = x + drift * dt
        x = x_mean + diffusion * np.sqrt(-dt) * z
        return x, x_mean

# langevin VE-SDE
class LangevinCorrector(Corrector):
    # 只用于VESDE
    def __init__(self, sde, snr, n_steps):
        super().__init__(sde, snr, n_steps)

    def update_fn(self, model, x, t, y):
        alpha = torch.ones_like(t)
        for i in range(self.n_steps):
            grad = self.sde.score_fn(model, x, t, y)  # [b ,c,h,w]
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha#[t->n 1 1 1]
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    def __init__(self, sde, snr, n_steps):
        super().__init__(sde, snr, n_steps)

    def update_fn(self, model, x, t, y):
        alpha = torch.ones_like(t)
        std = (self.sde.sigma ** t)

        for i in range(self.n_steps):
            grad = self.sde.score_fn(model, x, t, y)
            noise = torch.randn_like(x)
            step_size = ((self.snr * std) ** 2 * 2 * alpha)[:, None, None, None]#[t->n 1 1 1]
            x_mean = x + step_size * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)
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
        self.predictor = ReverseDiffusionPredictor(self.sde)
        self.corrector = LangevinCorrector(self.sde, 0.16, 1)

    def forward(self, x_0, label):
        self.batch = x_0.shape[0]
        # 随机生成不同img的时间步
        # # (0-1)*(1-0.00005)+0.00005
        t = torch.rand(self.batch, device=self.device) * (self.sde.T - self.eps) + self.eps
        noise = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, t)
        x_t = mean + std * noise
        # score_fn*std
        score = self.sde.score_fn(self.model, x_t, t, y=label) * std

        # loss = torch.mean(torch.sum(torch.square(score + noise).view(self.batch, -1), dim=-1))
        loss = self.criterion(score, noise)
        return loss

    @torch.no_grad()
    def sample(self, sample_num=16, labels=None):
        shape = (sample_num, self.channels, self.image_size, self.image_size)
        x = self.sde.prior_sampling(shape).to(self.device)
        timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)
        for i in tqdm(range(self.sde.N), desc="sample"):  # 1000
            t = timesteps[i]
            vec_t = torch.ones(sample_num, device=t.device) * t
            x, x_mean = self.corrector.update_fn(self.model, x, t=vec_t, y=labels)
            x, x_mean = self.predictor.update_fn(self.model, x, t=vec_t, y=labels)

        img = torch.clamp_(x_mean, min=-1, max=1)
        img = (img + 1) * 0.5
        return img
