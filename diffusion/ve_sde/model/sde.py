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

    @abc.abstractmethod
    def marginal_prob(self, x_0, t):
        """Compute the mean/std of the transitional kernel
        p(x_t | x_0).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def score_fn(self, model, x_t, t, y):
        raise NotImplementedError


class VESDE(AbstractSDE, torch.nn.Module):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        sigma = torch.tensor(self.sigma_min * (self.sigma_max / self.sigma_min))
        sqrt_delta_sima = torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))))
        self.register_buffer("sigmas",sigmas)
        self.register_buffer("sigma",sigma)
        self.register_buffer("sqrt_delta_sima",sqrt_delta_sima)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        drift = torch.zeros_like(x)
        diffusion = (self.sigma ** t) * self.sqrt_delta_sima
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = (self.sigma ** t)[:, None, None, None]
        mean = x
        return mean, std

    def score_fn(self, model, x_t, t, y):
        score = model(x_t, self.sigma ** t, y)
        return score

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), self.sigmas[timestep - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]  # [b]->[b,1,1,1]
        return f, G


    def reverse(self, model, x, t, y):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = self.discretize(x, t)
        rev_f = f - G ** 2 * self.score_fn(model, x, t, y)
        rev_G = G
        return rev_f, rev_G
