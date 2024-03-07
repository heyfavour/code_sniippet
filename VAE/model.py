import sys
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU())
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class VAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 128

        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.fc_mu = nn.Linear(512 * 4, 128)
        self.fc_var = nn.Linear(512 * 4, 128)

        self.decoder_input = nn.Linear(128, 512 * 4)
        self.decoder = nn.Sequential(
            ConvTransposeBlock(512, 256),
            ConvTransposeBlock(256, 128),
            ConvTransposeBlock(128, 64),
            ConvTransposeBlock(64, 32),
        )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def encode(self, input: Tensor) -> List[Tensor]:
        x = self.encoder(input)  # [b 3 64 64]->[b 512 2 2]
        x = torch.flatten(x, start_dim=1)  # [b 512 2 2] [b 2048]
        mu, log_var = self.fc_mu(x), self.fc_var(x)
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)  # [64 2048]
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x



    def loss_function(self, recons, input, mu, log_var, kld_weight) -> dict:
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return loss

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]
