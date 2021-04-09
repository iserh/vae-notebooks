"""Variational autoencoder module class."""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
import torch
from torch.optim import Adam


def init_weights(module: nn.Module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    elif type(module) == nn.Conv2d:
        nn.init.dirac_(module.weight)
    elif type(module) == nn.ConvTranspose2d:
        nn.init.dirac_(module.weight)


class Encoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.conv_stage = nn.Sequential(
            # input is (3) x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 16 x 16
            nn.Conv2d(64, 64 * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64 * 4),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64 * 8),
            # # state size. (64*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(64 * 8 * 4 * 4, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
        )
        # Encoder mean
        self.mean = nn.Linear(128, z_dim)
        # Encoder Variance log
        self.variance_log = nn.Linear(128, z_dim)

        # initialize weights
        self.conv_stage.apply(init_weights)
        self.mean.apply(init_weights)
        self.variance_log.apply(init_weights)

    def forward(self, x: Tensor):
        x = self.conv_stage(x)
        return self.mean(x), self.variance_log(x)


class Decoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.linear_stage = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.Linear(128, 64 * 8 * 4 * 4),
        )
        self.conv_stage = nn.Sequential(
            # # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 16 x 16
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            # state size. (3) x 32 x 32
            nn.Sigmoid(),
        )

        # initialize weights
        self.linear_stage.apply(init_weights)
        self.conv_stage.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_stage(x)
        x = x.view(x.size(0), 64 * 8, 4, 4)
        return self.conv_stage(x)


class VAE(pl.LightningModule):
    def __init__(self, input_size: int, z_dim: int, beta: float):
        super().__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        self.beta = beta
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
    
    def forward(self, x):
        # encode
        m, v_log = self.encoder(x)
        if self.training:
            # sample from encoding
            eps = torch.empty_like(v_log).normal_()
            z = eps * (v_log / 2).exp() + m
        else:
            z = m
        # decode from sample
        x_ = self.decoder(z)
        # return reconstruction, mean and log_variance
        return x_, m, v_log
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_, m, v_log = self(x)
        r_l = F.binary_cross_entropy(x_, x) * self.input_size
        kl_l = (v_log.exp() + m**2 - 1 - v_log).sum(-1).mean()
        loss = self.beta * kl_l + r_l
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4, weight_decay=3e-5)
