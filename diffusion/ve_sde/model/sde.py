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
    def scale_start_to_noise(self, t):
        """Compute the scale of conversion
        from the original image estimation loss, i.e, || x_0 - x_0_pred ||
        to the noise prediction loss, i.e, || e - e_pred ||.
        Denoting the output of this function by C,
        C * || x_0 - x_0_pred || = || e - e_pred || holds.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score_fn(self, model, x_t, t, y):
        raise NotImplementedError

    # @abc.abstractmethod
    # def proposal_distribution(self):
    #     raise NotImplementedError

    def reverse(self, model, x_t, t, y):
        score = self.score_fn(model, x_t, t, y)
        drift, diffusion = self.sde(x_t, t)
        drift = drift - diffusion ** 2 * score
        return drift, diffusion


class VPSDE(AbstractSDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.beta = torch.linspace(beta_min / N, beta_max / N, N)
        self.alpha = 1. - self.beta

        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x_t, t):
        beta_t = (self.beta_0 + t * (self.beta_1 - self.beta_0))[:, None, None, None]
        drift = -0.5 * beta_t * x_t
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        # coeff = (-0.25 * t ^2 * (20-0.05) - 1/2 * t * 0.05)
        log_mean_coeff = (-0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0)[:, None, None, None]
        mean = torch.exp(log_mean_coeff) * x_0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def score_fn(self, model, x_t, t, y):
        score = model(x_t, t * 999, y)
        std = torch.sqrt(
            1. - torch.exp(2. * -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0)
        )[:, None, None, None]
        score = - score / std
        return score

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = - N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def scale_start_to_noise(self, t):
        log_mean_coeff = (
                                 -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
                         )[:, None, None, None]
        marginal_coeff = torch.exp(log_mean_coeff)
        marginal_std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        scale = marginal_coeff / (marginal_std + 1e-12)
        return scale
