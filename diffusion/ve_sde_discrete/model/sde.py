import abc
import numpy as np
import torch


class AbstractSDE(abc.ABC):
    def __init__(self):
        super().__init__()
        self.N = 1000

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        raise NotImplementedError

    @abc.abstractmethod
    def sde(self, x_t, t):
        """Compute the drift/diffusion of the forward SDE
        dx = b(x_t, t)dt + s(x_t, t)dW
        """
        raise NotImplementedError

    def marginal_prob(self, x_0, t):
        #Compute the mean/std of the transitional kernel p(x_t | x_0). 平时用在continues
        raise NotImplementedError

    def prior_logp(self, z):
        """Compute log-density of the prior distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def score_fn(self, model, x_t, t, y):
        # VESDE VPSDE continues 4种区别
        raise NotImplementedError



class VESDE(AbstractSDE, torch.nn.Module):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        sigma = torch.tensor((self.sigma_max / self.sigma_min))
        sqrt_delta_sima = torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))))
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("sigma", sigma)
        self.register_buffer("sqrt_delta_sima", sqrt_delta_sima)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        drift = torch.zeros_like(x)
        diffusion = (self.sigma_min * self.sigma ** t) * self.sqrt_delta_sima
        return drift, diffusion[:, None, None, None]

    # VESED continues=False
    def score_fn(self, model, x_t, t, y):
        t = torch.round((self.T - t) * (self.N - 1)).long()#int((1-t)*999)
        score = model(x_t, t, y)
        return score

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def discretize(self, x, t):
        sigma = self.sigmas[t]
        adjacent_sigma = torch.where(t == 0, torch.zeros_like(t), self.sigmas[t - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]  # [b]->[b,1,1,1]
        return f, G




