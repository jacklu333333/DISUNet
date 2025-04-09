import numpy as np
import torch
from torch import nn, optim


class DDPM_Scheduler(nn.Module):
    def __init__(
        self, num_time_steps: int = 1000, modes: str = "linear", s=None, e=None
    ):
        super().__init__()
        if modes == "linear":
            if not s:
                s = 1e-4
            if not e:
                e = 0.02
            self.betas = torch.linspace(s, e, num_time_steps, requires_grad=False)
        elif modes == "exp":
            if not s:
                s = -12
            if not e:
                e = -2
            self.betas = torch.exp(
                torch.linspace(s, e, num_time_steps, requires_grad=False)
            )
        elif modes == "cosine":
            if not s:
                s = 1e-4
            if not e:
                e = 0.008
            steps = torch.arange(num_time_steps, dtype=torch.float32)
            self.betas = (
                0.5 * (1 + torch.sin(np.pi * steps / num_time_steps)) * (e - s) + s
            )
        elif modes == "quad":
            if not s:
                s = 1e-4
            if not e:
                e = 0.02
            self.betas = (
                torch.linspace(s**0.5, e**0.5, num_time_steps, requires_grad=False) ** 2
            )
        else:
            raise ValueError("Invalid scheduler mode")
        self.weight = torch.ones_like(self.betas[1:-1])
        self.alphas = (1 - self.betas) * torch.cat(
            [torch.ones(1), self.weight, torch.ones(1)]
        )
        self.alpha_products = torch.cumprod(self.alphas, dim=0)
        self.sigma = torch.sqrt(self.betas)

    def forward(self, t, batch_size, device):
        self.betas = self.betas.to(device)
        self.weight = nn.Parameter(
            torch.ones_like(self.betas[1:-1]), requires_grad=True
        ).to(device)
        self.alphas = (1 - self.betas) * torch.cat(
            [torch.ones(1, device=device), self.weight, torch.ones(1, device=device)]
        ).to(device)
        self.alpha_products = torch.cumprod(self.alphas, dim=0).to(device)

        self.sigma = torch.sqrt(self.betas).to(device)

        return (
            self.betas[t].view(batch_size, 1, 1),
            self.alpha_products[t].view(batch_size, 1, 1),
            self.alphas[t].view(batch_size, 1, 1),
            self.sigma[t].view(batch_size, 1, 1),
        )

    def naive_sample_forward(self, pseudo_noise, ground_truth, t):
        index = t < 0
        t = torch.clamp(t, min=0)
        beta, alpha_product, alpha, sigma = self(
            t, pseudo_noise.shape[0], pseudo_noise.device
        )
        noisy = pseudo_noise.clone().detach()
        noisy = (beta * pseudo_noise) + ((1 - beta) * ground_truth)
        if index.any():
            noisy[index] = ground_truth[index]

        return noisy

    def sample_forward(self, pseudo_noise, ground_truth, t):
        index = t < 0
        t = torch.clamp(t, min=0)
        beta, alpha_product, alpha, sigma = self(
            t, pseudo_noise.shape[0], pseudo_noise.device
        )
        noisy = pseudo_noise.clone().detach()
        noisy = (torch.sqrt(alpha_product) * ground_truth) + (
            torch.sqrt(1 - alpha_product) * pseudo_noise
        )
        if index.any():
            noisy[index] = ground_truth[index]

        return noisy

    def sample_backward(self, pseudo_observation, noise_estimated, t, z=None):
        beta, alpha_product, alpha, sigma = self(
            t, pseudo_observation.shape[0], pseudo_observation.device
        )
        index = t < 1
        z[index] = torch.zeros_like(z[index])

        if z is None:
            z = torch.zeros_like(pseudo_observation)
        denoisy = pseudo_observation.clone().detach()
        denoisy = (
            1
            / torch.sqrt(alpha)
            * (
                pseudo_observation
                - (1 - alpha) / torch.sqrt(1 - alpha_product) * noise_estimated
            )
            + sigma * z
        )

        return denoisy


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch, **kwargs):
        epoch = float(epoch)
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class WarmupReduceLROnPlateau(optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, reduce_factor, patience, min_lr=0, verbose=False
    ):
        self.warmup_epochs = warmup_epochs
        self.reduce_factor = reduce_factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.warmup_scheduler = None
        self.reduce_scheduler = None
        self.current_epoch = 0
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, verbose)

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return [
                base_lr * (self.current_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            if self.reduce_scheduler is None:
                self.reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=self.reduce_factor,
                    patience=self.patience,
                    min_lr=self.min_lr,
                    verbose=self.verbose,
                )
            return self.reduce_scheduler.optimizer.param_groups[0]["lr"]

    def step(self, metrics=None):
        self.current_epoch += 1
        if self.current_epoch > self.warmup_epochs:
            self.reduce_scheduler.step(metrics)
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
