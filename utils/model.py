import math
import os
from typing import List

import colored as cl
import deepspeed
import diffusers
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange

# from mixture_of_experts import HeirarchicalMoE, MoE
from tensorboard.backend.event_processing import event_accumulator
from timm.utils import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import (
    CosineSimilarity,
    KLDivergence,
    MeanAbsoluteError,
    MeanSquaredError,
)
from tqdm import tqdm

from utils.transform import (
    axisMasking,
    gaussianNoise,
    rotationNoise,
    scaleNoise,
    shiftNoise,
    trnasformBatch,
)

from .activation import limiterActivation
from .datasets import SCALE
from .loss import *
from .scheduler import CosineWarmupScheduler, DDPM_Scheduler


def extract_metrics(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()["scalars"]

    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        if norm:
            self.norm1 = nn.Sequential(
                Rearrange("b c l -> b l c"),
                nn.BatchNorm1d(hidden_dim),
                Rearrange("b l c -> b c l"),
            )
            self.norm2 = nn.Sequential(
                Rearrange("b c l -> b l c"),
                nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
                Rearrange("b l c -> b c l"),
            )

    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.fc2(x)
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.fc3(x)
        return x


class sinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int, in_channels: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

        self.mlp = MLP(in_channels + embed_dim, embed_dim, embed_dim, norm=True)

    def forward(self, x, t):
        embeds = self.embeddings.to(x.device)
        embeds = embeds[t].to(x.device).unsqueeze(1)
        result = torch.cat([x, embeds.expand(-1, x.shape[1], -1)], dim=-1)
        result = self.mlp(result)
        return result


class positionEncoding(nn.Module):
    def __init__(self, length: int, embed_dim: int, in_channels: int):
        super().__init__()
        self.embeddings = torch.zeros((length, embed_dim), requires_grad=False)
        pos = torch.arange(0, length, dtype=torch.float).unsqueeze(dim=1)
        _2i = torch.arange(0, embed_dim, step=2, dtype=torch.float)
        self.embeddings[:, 0::2] = torch.sin(pos / 10000 ** (_2i / embed_dim))
        self.embeddings[:, 1::2] = torch.cos(pos / 10000 ** (_2i / embed_dim))

        self.mlp = MLP(in_channels + embed_dim, embed_dim, embed_dim, norm=True)

    def forward(self, x):
        result = torch.cat(
            [x, self.embeddings.expand(x.shape[0], -1, -1).to(x.device)], dim=-1
        )
        result = self.mlp(result)

        return result


class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout: float):
        super().__init__()
        self.activation = nn.GELU()
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)

        self.conv1 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, : x.shape[1], :]
        r = self.conv1(self.activation(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.activation(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        l = x.shape[-1]
        x = rearrange(x, "b c l -> b l c")
        x = self.proj1(x)
        x = rearrange(x, "b L (C H K) -> K b H L C", K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout_prob
        )
        x = rearrange(x, "b H l C -> b l (C H)", l=l)
        x = self.proj2(x)
        return rearrange(x, "b l C -> b C l")


class UnetLayer(nn.Module):
    def __init__(
        self,
        upscale: bool,
        attention: bool,
        num_groups: int,
        dropout_prob: float,
        num_heads: int,
        C: int,
        size: int = 1000,
    ):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout=dropout_prob)
        if upscale:
            self.conv = nn.Sequential(
                nn.ConvTranspose1d(C, C // 2, kernel_size=4, stride=2, padding=1),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(C, C * 2, kernel_size=3, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
        if attention:
            self.attention_layer = Attention(
                C, num_heads=num_heads, dropout_prob=dropout_prob
            )
            self.moe_layer = MoE(
                dim=size,
                num_experts=size * 2,
                hidden_dim=num_heads * 4,
                activation=nn.GELU,
                second_policy_train="all",
                second_policy_eval="all",
                second_threshold_train=0.2,
                second_threshold_eval=0.2,
                capacity_factor_train=1.25,
                capacity_factor_eval=2.0,
                loss_coef=1e-2,
            )

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, "attention_layer"):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        if hasattr(self, "attention_layer"):
            x, aux_loss = self.moe_layer(x)
        return self.conv(x), x, aux_loss


class CustomPadding(nn.Module):
    def __init__(self, padding_size, mode="replicate"):
        super(CustomPadding, self).__init__()
        self.padding_size = padding_size
        self.mode = mode

    def forward(self, x):
        return F.pad(x, self.padding_size, mode=self.mode)


class UNET(nn.Module):
    def __init__(
        self,
        Channels: List = [64, 128, 256, 512, 512, 384],
        Attentions: List = [True, True, True, True, True, True],
        Upscales: List = [False, False, False, True, True, True],
        num_groups: int = 32,
        dropout: float = 0.1,
        num_heads: int = 8,
        input_channels: int = 9,
        output_channels: int = 9,
        time_steps: int = 1000,
    ):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv1d(
            input_channels, Channels[0], kernel_size=3, padding=1
        )
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv1d(
            out_channels, out_channels // 2, kernel_size=3, padding=1
        )

        self.output_conv = nn.Sequential(
            nn.Conv1d(out_channels // 2, output_channels, kernel_size=1),
        )

        self.activation = nn.GELU()
        self.embeddings = sinusoidalEmbeddings(
            time_steps=time_steps, embed_dim=max(Channels)
        )
        s = [512, 256, 128, 64, 128, 256]
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout,
                C=Channels[i],
                num_heads=num_heads,
                size=s[i],
            )
            setattr(self, f"Layer{i+1}", layer)

    def forward(self, x, t, dataL):
        batch_size, channels, length = x.shape
        x = self.shallow_conv(x)
        residuals = []
        aux_losses = 0
        for i in range(self.num_layers // 2):
            layer = getattr(self, f"Layer{i+1}")
            embeddings = self.embeddings(x, t)
            x, r, aux_loss = layer(x, embeddings)
            aux_losses += aux_loss
            residuals.append(r)

        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f"Layer{i+1}")
            x_, r, aux_loss = layer(x, embeddings)
            aux_losses += aux_loss
            x = torch.concat(
                (
                    x_,
                    residuals[self.num_layers - i - 1],
                ),
                dim=1,
            )
        result = self.output_conv(self.activation(self.late_conv(x)))

        mask = self.mask_generation(dataL, length, batch_size)
        if mask.any():
            result = result.masked_fill(
                mask.unsqueeze(1).expand(-1, result.shape[1], -1), 0
            )
        return result, aux_losses

    def _get_regularization_params(self):
        params = []
        for module in self.modules():
            if isinstance(
                module, nn.Module
            ):  # Ensure the module is a valid PyTorch module
                for param in module.parameters():
                    params.append(param.view(-1))
        return torch.cat(params) if params else torch.tensor([])

    def initialize_weight_my_fucking_way(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            for submodule in module.children():
                self.initialize_weight_my_fucking_way(submodule)

    def mask_generation(self, dataL, length, batch_size):
        mask = torch.arange(length).expand(batch_size, length).to(dataL.device)
        mask = mask < dataL.unsqueeze(1)
        mask = torch.logical_not(mask)
        return mask


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        num_steps,
        length,
        embed_dim=16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.length = length
        self.embed_dim = embed_dim

        self.Project = nn.Sequential(
            Rearrange("b c l -> b l c"),
        )

        self.motion_moe = HeirarchicalMoE(
            dim=embed_dim,
            num_experts=(8, 8),
            activation=nn.ReLU,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation=F.relu,
            batch_first=True,
            norm_first=False,
        )
        normalization = nn.Sequential(
            nn.LayerNorm(embed_dim),
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            norm=normalization,
            num_layers=self.num_layers,
        )
        self.deProject = nn.Sequential(
            nn.Linear(embed_dim, self.out_channels),
            Rearrange("b l c -> b c l"),
        )

        self.step_embeddings = sinusoidalEmbeddings(
            time_steps=num_steps,
            embed_dim=embed_dim,
            in_channels=self.in_channels,
        )

        self.position_embedding = positionEncoding(
            length=length,
            embed_dim=embed_dim,
            in_channels=self.in_channels,
        )

    def mask_generation(self, dataL, length, batch_size):
        mask = torch.arange(length).expand(batch_size, length).to(dataL.device)
        mask = mask < dataL.unsqueeze(1)
        mask = torch.logical_not(mask)
        return mask

    def forward(self, x, t, dataL):
        batch_size, channels, length = x.shape
        dataL = dataL.to(x.device)
        mask = self.mask_generation(dataL, length, batch_size)

        observation = self.Project(x)

        motion = self.position_embedding(observation)
        if mask.any():
            motion.masked_fill(mask.unsqueeze(2).expand(-1, -1, self.embed_dim), 0)
        motion, aux1 = self.motion_moe(motion)

        step_observation = self.step_embeddings(observation, t)
        if mask.any():
            step_observation.masked_fill(
                mask.unsqueeze(2).expand(-1, -1, self.embed_dim), 0
            )

        real = self.decoder(
            tgt=step_observation,
            memory=motion,
            tgt_key_padding_mask=mask if mask.any() else None,
            memory_key_padding_mask=mask if mask.any() else None,
        )
        real = self.deProject(real)
        if mask.any():
            mask = mask.unsqueeze(1).expand(-1, real.shape[1], -1)
            real = real.masked_fill(mask, 0)

        return real, aux1

    def _get_regularization_params(self):
        params = []
        for module in self.modules():
            if isinstance(
                module, nn.Module
            ):  # Ensure the module is a valid PyTorch module
                for param in module.parameters():
                    params.append(param.view(-1))
        return torch.cat(params) if params else torch.tensor([])


class Acc_Vel_Pos(nn.Module):
    def __init__(self, num_layers, num_steps, lentgh, embed_dim=16, *args, **kwargs):
        super().__init__()
        self.acc_model = TransformerPredictor(
            num_layers=num_layers,
            num_steps=num_steps,
            length=lentgh,
        )
        self.vel_model = TransformerPredictor(
            num_layers=num_layers,
            num_steps=num_steps,
            length=lentgh,
        )

    def forward(self, x, y):
        acc, aux1 = self.acc_model(x)
        vel, aux2 = self.vel_model(acc.clone().cumsum(dim=-1))

        return acc, vel, aux1 + aux2


class dummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            Rearrange("b c l -> b l c"),
            nn.Linear(3, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3),
            nn.Tanh(),
            Rearrange("b l c -> b c l"),
        )

    def forward(self, sample, timestep, return_dict):
        x = self.fc1(sample)
        return [x]


class DiffusionModelPL(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(DiffusionModelPL, self).__init__()
        self.config = config

        self.transform = trnasformBatch(self.config["augmentation"])

        self.model = diffusers.models.UNet1DModel(
            sample_size=self.config["window_size"],
            sample_rate=100,
            in_channels=6,
            out_channels=6,
            extra_in_channels=0,
            time_embedding_type="fourier",
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            freq_shift=1.0,
            down_block_types=(
                "DownBlock1D",
                "AttnDownBlock1D",
                "DownBlock1D",
                "AttnDownBlock1D",
                "DownBlock1D",
                "AttnDownBlock1D",
            ),
            up_block_types=(
                "AttnUpBlock1D",
                "UpBlock1D",
                "AttnUpBlock1D",
                "UpBlock1D",
                "AttnUpBlock1D",
                "UpBlock1D",
            ),
            out_block_type=None,
            block_out_channels=(
                64,
                64,
                128,
                128,
                256,
                256,
            ),
            act_fn="mish",
            layers_per_block=3,
            downsample_each_block=True,
        )

        self.scheduler = DDPM_Scheduler(
            num_time_steps=config["num_time_steps"],
            s=config["scheduler_s"],
            e=config["scheduler_e"],
            modes=config["scheduler_mode"],
        )

        self.ema = ModelEmaV3(
            self.model,
            decay=config["ema_decay"],
        )

        self.loss = myspecialLoss(self.config["loss"])

        self.lr = self.config["lr"]

        self.metrics = {
            "train": self._generate_metrics("train"),
            "val": self._generate_metrics("val"),
            "test": self._generate_metrics("test"),
        }

        self.save_hyperparameters(self.config)
        self.df = None

    def mask_generation(self, dataL, length, batch_size):
        mask = torch.arange(length).expand(batch_size, length).to(dataL.device)
        mask = mask < dataL.unsqueeze(1)
        mask = torch.logical_not(mask)
        return mask

    def on_train_start(self) -> None:
        return super().on_train_start()

    def _generate_metrics(self, suffix):
        metrics = {
            f"metric_acc_pearson_X/{suffix}": PearsonMetric_acc_X(),
            f"metric_acc_pearson_Y/{suffix}": PearsonMetric_acc_Y(),
            f"metric_acc_pearson_Z/{suffix}": PearsonMetric_acc_Z(),
            f"metric_acc_pearson_norm/{suffix}": PearsonMetric_acc_Norm(),
            f"metric_acc_simVector/{suffix}": CosSimMetric_acc(),
            f"metric_acc_simVector_X/{suffix}": CosSimMetric_acc_X(),
            f"metric_acc_simVector_Y/{suffix}": CosSimMetric_acc_Y(),
            f"metric_acc_simVector_Z/{suffix}": CosSimMetric_acc_Z(),
            f"metric_acc_simVector_norm/{suffix}": CosSimMetric_acc_Norm(),
            f"metric_naive_distance_error_X/{suffix}": NaiveDistanceError_X(),
            f"metric_naive_distance_error_Y/{suffix}": NaiveDistanceError_Y(),
            f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError_Z(),
            f"metric_naive_distance_error_XY/{suffix}": NaiveDistanceError_XY(),
            f"metric_naive_distance_error/{suffix}": NaiveDistanceError(),
            f"metric_gyr_pearson_X/{suffix}": PearsonMetric_gyr_X(),
            f"metric_gyr_pearson_Y/{suffix}": PearsonMetric_gyr_Y(),
            f"metric_gyr_pearson_Z/{suffix}": PearsonMetric_gyr_Z(),
            f"metric_gyr_pearson_norm/{suffix}": PearsonMetric_gyr_Norm(),
            f"metric_gyr_simVector/{suffix}": CosSimMetric_gyr(),
            f"metric_gyr_simVector_X/{suffix}": CosSimMetric_gyr_X(),
            f"metric_gyr_simVector_Y/{suffix}": CosSimMetric_gyr_Y(),
            f"metric_gyr_simVector_Z/{suffix}": CosSimMetric_gyr_Z(),
            f"metric_gyr_simVector_norm/{suffix}": CosSimMetric_gyr_Norm(),
            f"metric_naive_Angular_error_X/{suffix}": NaiveAngularError_X(),
            f"metric_naive_Angular_error_Y/{suffix}": NaiveAngularError_Y(),
            f"metric_naive_Angular_error_Z/{suffix}": NaiveAngularError_Z(),
            f"metric_naive_Angular_error/{suffix}": NaiveAngularError(),
        }
        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics

    def forward(self, x):
        raise NotImplementedError

    def find_pad_size(self, x):
        pad_size = 0
        original_size = x.size(-1)
        power = np.ceil(np.log2(original_size))
        new_size = 2**power
        pad_size = int((new_size - original_size) // 2)

        return 0

    def sample_noise(self, batch, pad_size, dataL, do_noise=True):
        result = self.transform(batch) if do_noise else batch
        x, y = result
        x, y = x.clone().detach(), y.clone().detach()

        x, y = self.preprocessing((x, y, dataL))

        return x, y

    @torch.no_grad()
    def baseline_testing(self, batch):
        epsilon, x, dataL = batch
        epsilon, x = self.sample_noise((epsilon, x), 0, dataL=dataL, do_noise=False)
        estimate = epsilon
        estimate, x = self.postprocessing((estimate, x, dataL))

        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(estimate, x)

        self.log_dict(
            metrics,
            sync_dist=True,
        )

        return estimate

    @torch.no_grad()
    def naive_seq_sampling(self, batch):
        epsilon, x, dataL = batch
        batch_size, channels, length = x.shape
        pad_size = self.find_pad_size(x)

        times = torch.arange(
            start=self.config["num_time_steps"] - 1,
            end=0,
            step=-1,
            device=x.device,
        )

        epsilon_pad, x_pad = self.sample_noise(
            (epsilon, x), pad_size, dataL=dataL, do_noise=False
        )

        pseudo_observation = epsilon_pad.clone().detach()
        for t in tqdm(times, total=len(times)):
            estimate = self.model(
                sample=pseudo_observation,
                timestep=t,
                return_dict=False,
            )[0]
            mask = self.mask_generation(dataL, x.shape[-1], batch_size)
            if torch.any(mask):
                estimate = estimate.masked_fill(
                    mask.unsqueeze(1).expand(-1, estimate.shape[1], -1), 0
                )
            pseudo_observation = estimate

        estimate, x_pad = self.postprocessing((estimate, x_pad, dataL))

        estimate = estimate[:, :, pad_size : self.config["window_size"] + pad_size]
        x_pad = x_pad[:, :, pad_size : self.config["window_size"] + pad_size]

        metrics = {}
        for key, metric in self.metrics["test"].items():
            # metrics[key] = metric(estimate, x_pad)
            ms = metric(estimate, x_pad)
            for k, v in ms.items():
                metrics[f"{key}_{k}"] = v

        self.log_dict(
            metrics,
            sync_dist=True,
        )
        return estimate

    @torch.no_grad()
    def seq_sampling(self, batch):
        epsilon, x, dataL = batch
        batch_size, channels, length = x.shape
        pad_size = self.find_pad_size(x)

        times = torch.arange(
            start=self.config["num_time_steps"] - 1,
            end=-1,
            step=-1,
            device=x.device,
        )

        epsilon_pad, x_pad = self.sample_noise(
            (epsilon, x), pad_size, dataL=dataL, do_noise=False
        )

        pseudo_observation = epsilon_pad.clone().detach()
        for t in tqdm(times, total=len(times)):
            estimate, _ = self.sampling(
                pseudo_observation=pseudo_observation,
                seed=epsilon_pad,
                t=torch.tensor([t] * x.shape[0], device=x.device),
                dataL=dataL,
            )
            pseudo_observation[:, :3] = estimate[:, :3]
        estimate, x_pad = self.postprocessing((estimate, x_pad, dataL))
        estimate = estimate[:, :3]

        estimate = estimate[:, :, pad_size : self.config["window_size"] + pad_size]
        x_pad = x_pad[:, :, pad_size : self.config["window_size"] + pad_size]

        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(estimate, x_pad)

        self.log_dict(
            metrics,
            sync_dist=True,
        )
        return estimate

    @torch.no_grad()
    def sampling(self, pseudo_observation, t, seed, dataL):
        z = seed.clone().detach()

        noise_estimated = self.model(
            sample=pseudo_observation[:, :3],
            timestep=t,
            return_dict=False,
        )[0]
        mask = self.mask_generation(
            dataL, pseudo_observation.shape[-1], pseudo_observation.shape[0]
        )
        if torch.any(mask):
            noise_estimated = noise_estimated.masked_fill(
                mask.unsqueeze(1).expand(-1, noise_estimated.shape[1], -1), 0
            )

        if torch.all(noise_estimated == 0):
            print(cl.fg("red"), "All the noise are zeros", cl.attr("reset"))
        x_t_minus_1 = self.scheduler.sample_backward(
            pseudo_observation=pseudo_observation,
            noise_estimated=noise_estimated,
            t=t,
            z=z,
        )

        return x_t_minus_1, noise_estimated

    def clamp_tensor(self, x_t_minus_1, original, dataL):
        min_val = original.min(dim=-1, keepdim=True).values
        max_val = original.max(dim=-1, keepdim=True).values
        x_t_minus_1 = (
            torch.cat(
                [torch.zeros_like(x_t_minus_1[:, :, :1]), x_t_minus_1], dim=-1
            ).diff(dim=-1)
            / 0.01
        )
        x_t_minus_1 = torch.clamp(x_t_minus_1, min_val, max_val)
        x_t_minus_1 = x_t_minus_1.cumsum(dim=-1) * 0.01

        batch_size, channels, length = x_t_minus_1.shape
        mask = self.mask_generation(dataL, length, batch_size)
        if torch.any(mask):
            x_t_minus_1 = x_t_minus_1.masked_fill(
                mask.unsqueeze(1).expand(-1, x_t_minus_1.shape[1], -1), 0
            )
        return x_t_minus_1

    def scaler_tensor(self, x_t_minus_1, original, dataL):
        std_original = original.std(dim=-1, keepdim=True)
        new_std = x_t_minus_1.std(dim=-1, keepdim=True)

        x_t_minus_1 = x_t_minus_1 * (std_original / new_std)
        return x_t_minus_1

    def preprocessing(self, batch):
        x, y, dataL = batch
        batch_size, channels, length = x.shape
        pad_size = self.find_pad_size(x)

        x_, y_ = self.transform((x, y))

        mask = self.mask_generation(dataL, length, batch_size)
        if torch.any(mask):
            x_ = x_.masked_fill(mask.unsqueeze(1).expand(-1, x_.shape[1], -1), 0)
            y_ = y_.masked_fill(mask.unsqueeze(1).expand(-1, y_.shape[1], -1), 0)

        x_ = F.pad(x_, (pad_size, pad_size), "constant", 0)
        y_ = F.pad(y_, (pad_size, pad_size), "constant", 0)
        return x_, y_

    def postprocessing(self, batch):
        x, y, dataL = batch
        batch_size, channels, length = x.shape
        x_, y_ = x, y

        return x_, y_

    def _get_regularization_params(self):
        params = []
        for module in self.modules():
            if isinstance(
                module, nn.Module
            ):  # Ensure the module is a valid PyTorch module
                for param in module.parameters():
                    params.append(param.view(-1))
        return torch.cat(params) if params else torch.tensor([])

    def naive_forward(self, batch, mode):
        epsilon, x, dataL = batch
        batch_size = x.shape[0]
        pad_size = self.find_pad_size(x)

        epsilon_pad, x_pad = self.sample_noise(
            (epsilon, x),
            pad_size,
            dataL=dataL,
            do_noise=True if mode == "train" else False,
        )

        t = torch.randint(
            1,
            self.config["num_time_steps"],
            (batch_size,),
            requires_grad=False,
            device=x.device,
        )
        t_prime = t.clone().detach()
        t_prime -= 1

        target = self.scheduler.naive_sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t_prime,
        )

        pseudo_observation = self.scheduler.naive_sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )

        output = self.model(
            sample=pseudo_observation,
            timestep=t,
            return_dict=False,
        )[0]
        mask = self.mask_generation(dataL, x.shape[-1], batch_size)
        if torch.any(mask):
            output = output.masked_fill(
                mask.unsqueeze(1).expand(-1, output.shape[1], -1), 0
            )
        aux_loss = F.l1_loss(
            torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
        )

        l1_loss = torch.norm(self._get_regularization_params(), 1)
        l2_loss = torch.norm(self._get_regularization_params(), 2)

        losses = self.loss(
            x=output,
            y=target,
            restoration=output,
            l1=l1_loss,
            l2=l2_loss,
            pad_size=pad_size,
            aux_loss=aux_loss,
            mode=mode,
        )
        noise_mse = F.mse_loss(output, epsilon_pad)

        output = output[
            :, : output.shape[1], pad_size : pad_size + self.config["window_size"]
        ]
        x_pad = x_pad[
            :, : output.shape[1], pad_size : pad_size + self.config["window_size"]
        ]

        metrics = {}
        if mode in self.metrics.keys():
            for key, metric in self.metrics[mode].items():
                # metrics[key] = metric(output, target)
                ms = metric(output, x_pad)
                for k, m in ms.items():
                    metrics[f"{key}_{k}"] = m

        self.log(
            f"noise_mse/{mode}",
            noise_mse,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            losses,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            metrics,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
        )

        return output, epsilon_pad, losses[f"loss/{mode}"]

    def special_forward(self, batch, mode):
        # x, y = batch
        epsilon, x, dataL = batch
        batch_size = x.shape[0]
        pad_size = self.find_pad_size(x)
        epsilon_pad, x_pad = self.sample_noise(
            (epsilon, x),
            pad_size,
            dataL=dataL,
            do_noise=True if mode == "train" else False,
        )

        t = torch.randint(
            0,
            self.config["num_time_steps"],
            (batch_size,),
            requires_grad=False,
            device=x.device,
        )

        noise_input = self.scheduler.sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )

        assert torch.isfinite(noise_input).all()
        if torch.all(noise_input == 0):
            print(cl.fg("red"), "All the noise are zeros", cl.attr("reset"))

        output = self.model(
            sample=noise_input[:, :3],
            timestep=t,
            return_dict=False,
        )[0]
        mask = self.mask_generation(dataL, x.shape[-1], batch_size)
        if torch.any(mask):
            output = output.masked_fill(
                mask.unsqueeze(1).expand(-1, output.shape[1], -1), 0
            )
        aux_loss = F.l1_loss(
            torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
        )

        restoration_t_minus_1 = self.scheduler.sample_backward(
            pseudo_observation=noise_input[:, :3],
            noise_estimated=output,
            t=t,
            z=self.sample_noise((epsilon, x), pad_size, dataL)[0],
        )
        l1_loss = torch.norm(self._get_regularization_params(), 1)
        l2_loss = torch.norm(self._get_regularization_params(), 2)

        t_minus_1 = self.scheduler.sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )[:, :3]
        losses = self.loss(
            x=output,
            y=epsilon_pad[:, :3],
            restoration=restoration_t_minus_1,
            l1=l1_loss,
            l2=l2_loss,
            pad_size=pad_size,
            aux_loss=aux_loss,
            mode=mode,
        )
        noise_mse = F.mse_loss(output, epsilon_pad[:, :3])

        restoration_t_minus_1, x_pad = self.postprocessing(
            (restoration_t_minus_1, x_pad, dataL)
        )
        restoration_t_minus_1 = restoration_t_minus_1[
            :, :3, pad_size : pad_size + self.config["window_size"]
        ]
        x_pad = x_pad[:, :3, pad_size : pad_size + self.config["window_size"]]

        metrics = {}
        if mode in self.metrics.keys():
            for key, metric in self.metrics[mode].items():
                metrics[key] = metric(restoration_t_minus_1, x_pad)

        self.log(
            f"noise_mse/{mode}",
            noise_mse,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            losses,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            metrics,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
        )

        return restoration_t_minus_1, output, epsilon_pad, losses[f"loss/{mode}"]

    def rescaling(self, x, y, dataL):

        df = self.df
        batch_size, channels, length = x.shape
        mask = self.mask_generation(dataL, length, batch_size)

        x_ = x.clone().detach()
        x_[:, 0:3] = (
            (x_[:, 0:3] - df["acc_mean"][0])
            / df["acc_std"][0]
            / SCALE[self.config["dataset"]]["acc"]
        )
        x_[:, 3:6] = (
            (x_[:, 3:6] - df["gyr_mean"][0])
            / df["gyr_std"][0]
            / SCALE[self.config["dataset"]]["gyr"]
        )

        y_ = (
            (y[:, 0:3].clone().detach() - df["l_mean"][0])
            / df["l_std"][0]
            / SCALE[self.config["dataset"]]["label"]
        )

        x_ = torch.clamp(x_, -1, 1)
        y_ = torch.clamp(y_, -1, 1)
        if torch.any(mask):
            x_ = x_.masked_fill(mask.unsqueeze(1).expand(-1, x.shape[1], -1), 0)
            y_ = y_.masked_fill(mask.unsqueeze(1).expand(-1, y.shape[1], -1), 0)

        if torch.all(y_ == 0):
            print(cl.fg("red"), "All the y are zeros", cl.attr("reset"))
        return x_, y_

    def descaling(self, x, y, dataL):
        batch_size, channels, length = x.shape
        mask = self.mask_generation(dataL, length, batch_size)
        df = self.df
        x_ = x.clone().detach()
        y_ = y[:, 0:3].clone().detach()

        x_[:, 0:3] = (
            x_[:, 0:3] * SCALE[self.config["dataset"]]["label"] * df["l_std"][0]
            + df["l_mean"][0]
        )
        x_[:, 3:6] = (
            x_[:, 3:6] * SCALE[self.config["dataset"]]["gyr"] * df["gyr_std"][0]
            + df["gyr_mean"][0]
        )

        y_ = (
            y_ * SCALE[self.config["dataset"]]["label"] * df["l_std"][0]
            + df["l_mean"][0]
        )
        if torch.any(mask):
            x_ = x_.masked_fill(mask.unsqueeze(1).expand(-1, x.shape[1], -1), 0)
            y_ = y_.masked_fill(mask.unsqueeze(1).expand(-1, y.shape[1], -1), 0)

        return x_, y_

    def normalize(self, x, y):
        acc = x[:, 0:3]
        gyr = x[:, 3:6]
        label = y[:, :3]

        mean_acc = x.mean(dim=-1, keepdim=True)
        std_acc = x.std(dim=-1, keepdim=True)
        mean_gyr = x.mean(dim=-1, keepdim=True)
        std_gyr = x.std(dim=-1, keepdim=True)
        mean_label = y.mean(dim=-1, keepdim=True)
        std_label = y.std(dim=-1, keepdim=True)

        acc = (acc - mean_acc) / std_acc
        gyr = (gyr - mean_gyr) / std_gyr

        label = (label - mean_label) / std_label

        return torch.cat([acc, gyr], dim=1), label

    def on_train_start(self):
        for case in ["train", "val", "test"]:
            self.loss.clear_history(mode=case)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        output, epsilon, total_loss = self.naive_forward(batch, "train")

        return total_loss

    def limit_range(
        self,
    ):
        pass

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        self.ema.update(self.model)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        for case in ["train", "val", "test"]:
            self.loss.clear_history(mode=case)
        return super().on_train_epoch_end()

    def on_train_end(self) -> None:
        self.limit_range()
        return super().on_train_end()

    def validation_step(self, batch, batch_idx):
        output, epsilon, total_loss = self.naive_forward(batch, "val")

        return total_loss

    def on_validation_end(self) -> None:
        self.limit_range()
        return super().on_validation_end()

    def test_step(self, batch, batch_idx):
        estimate = self.naive_seq_sampling(batch=batch)
        return torch.nan

    def on_test_end(self):
        self.limit_range()
        if os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0":
            log_dir = self.logger.log_dir
            metrics = extract_metrics(log_dir)

            log_dir = log_dir.split("/")
            log_dir = "/".join(log_dir[:-1])
            log_dir = log_dir + "/baseline"

            if not os.path.exists(log_dir):
                print(cl.Fore.red, "Baseline does not exist", cl.Style.reset)
                return super().on_test_end()
            baseline_metrics = extract_metrics(log_dir)

            config = self.config.copy()

            for key, value in metrics.items():
                if key not in baseline_metrics or (
                    ("pearson" not in key)
                    and ("simVector" not in key)
                    and ("naive" not in key)
                ):
                    continue
                m = value[-1][1]
                bm = baseline_metrics[key][-1][1]

                positive_relation = True
                coefficient = 1
                if "naive" in key:
                    positive_relation = False
                    coefficient = -1

                if (m > bm) == positive_relation:
                    word = "^v^"
                    COLOR = cl.Fore.red
                else:
                    word = "@A@"
                    COLOR = cl.Fore.green
                print(
                    COLOR,
                    f"{self.config['dataset']:5s} {word:5s} {key:40s} {m:>7.4f} with Gain of {m-bm:>7.4f} and  basline {bm:>7.4f}",
                    cl.Style.reset,
                )
        return super().on_test_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = 10
        times = torch.randint(0, self.config["num_time_steps"], (sample,))
        seq = []
        x, y = batch
        seq.append(x)
        for t in reversed(range(1, self.config["num_time_steps"])):
            restoation, output, epsilon, total_loss = self.special_forward(
                (seq[-1], y), "pred"
            )
            assert torch.all(torch.isfinite(output)), "output is not finite"

            if t in times:
                seq.append(output)

        return seq

    def configure_optimizers(self):
        self.config["lr"] = self.lr
        self.save_hyperparameters(self.config)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineWarmupScheduler(optimizer, warmup=50, max_iters=250)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train_step",
                "interval": "step",
                "frequency": 1,
            },
        }


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_regularization_params(self):
        params = []
        for module in self.modules():
            if isinstance(
                module, nn.Module
            ):  # Ensure the module is a valid PyTorch module
                for param in module.parameters():
                    params.append(param.view(-1))
        return torch.cat(params) if params else torch.tensor([])


class CycleGAN(pl.LightningModule):
    def __init__(self, config):
        super(CycleGAN, self).__init__()
        self.config = config
        self.generatorAcc = DiffusionModelPL(config)
        self.generatorMea = DiffusionModelPL(config)
        self.discriminatorAcc = Discriminator(6, 64, 1)
        self.discriminatorMea = Discriminator(6, 64, 1)

    def forward(self, x):
        raise NotImplementedError


class baseTrainingModel(DiffusionModelPL):
    def special_forward(self, batch, mode):
        epsilon, x, dataL = batch
        batch_size = x.shape[0]
        pad_size = self.find_pad_size(x)
        epsilon_pad, x_pad = self.sample_noise(
            (epsilon, x),
            pad_size,
            dataL=dataL,
            do_noise=True if mode == "train" else False,
        )

        t = torch.randint(
            0,
            self.config["num_time_steps"],
            (batch_size,),
            requires_grad=False,
            device=x.device,
        )

        noise_input = self.scheduler.sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )

        assert torch.isfinite(noise_input).all()
        if torch.all(noise_input == 0):
            print(cl.fg("red"), "All the noise are zeros", cl.attr("reset"))

        output = self.model(
            sample=noise_input[:, :3],
            timestep=t,
            return_dict=False,
        )[0]
        mask = self.mask_generation(dataL, x.shape[-1], batch_size)
        if torch.any(mask):
            output[mask.unsqueeze(1).expand(-1, output.shape[1], -1)] = 0
        aux_loss = F.l1_loss(
            torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
        )

        restoration_t_minus_1 = self.scheduler.sample_backward(
            pseudo_observation=noise_input[:, :3],
            noise_estimated=output,
            t=t,
            z=self.sample_noise((epsilon, x), pad_size, dataL)[0],
        )
        l1_loss = torch.norm(self._get_regularization_params(), 1)
        l2_loss = torch.norm(self._get_regularization_params(), 2)

        t_minus_1 = self.scheduler.sample_forward(
            pseudo_noise=epsilon_pad,
            ground_truth=x_pad,
            t=t,
        )[:, :3]
        losses = self.loss(
            x=output,
            y=epsilon_pad[:, :3],
            restoration=restoration_t_minus_1,
            l1=l1_loss,
            l2=l2_loss,
            pad_size=pad_size,
            aux_loss=aux_loss,
            mode=mode,
        )
        noise_mse = F.mse_loss(output, epsilon_pad[:, :3])

        restoration_t_minus_1, x_pad = self.postprocessing(
            (restoration_t_minus_1, x_pad, dataL)
        )
        restoration_t_minus_1 = restoration_t_minus_1[
            :, :3, pad_size : pad_size + self.config["window_size"]
        ]
        x_pad = x_pad[:, :3, pad_size : pad_size + self.config["window_size"]]

        metrics = {}
        if mode in self.metrics.keys():
            for key, metric in self.metrics[mode].items():
                metrics[key] = metric(restoration_t_minus_1, x_pad)

        self.log(
            f"noise_mse/{mode}",
            noise_mse,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            losses,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            metrics,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
        )

        return restoration_t_minus_1, output, epsilon_pad, losses[f"loss/{mode}"]


def batchStepBatch(scheduler, original, noise, t):
    batch_size = original.shape[0]
    result = original.clone().detach()[:, :3]
    for i in range(batch_size):
        result[i] = scheduler.step(
            model_output=noise[i],
            timestep=t[i],
            sample=original[i],
        ).prev_sample

    return result


class baseDiffusionModule(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.lr = self.config["lr"]
        # self.warm_up = 100
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            num_train_timesteps=self.config["num_time_steps"],
            beta_start=self.config["scheduler_s"],
            beta_end=self.config["scheduler_e"],
            beta_schedule=self.config["scheduler_mode"],
        )
        self.scheduler.set_timesteps(config["num_time_steps"])
        self.transform = trnasformBatch(self.config["augmentation"])

        self.metrics = {
            "train": self._generate_metrics("train"),
            "val": self._generate_metrics("val"),
            "test": self._generate_metrics("test"),
        }

        self.save_hyperparameters(self.config)

    def preprocessing(self, batch):
        x, y, dataL = batch
        batch_size, channels, length = x.shape
        pad_size = self.find_pad_size(x)

        x_, y_ = self.transform((x, y))

        mask = self.mask_generation(dataL, length, batch_size)
        if torch.any(mask):
            x_ = x_.masked_fill(mask.unsqueeze(1).expand(-1, x_.shape[1], -1), 0)
            y_ = y_.masked_fill(mask.unsqueeze(1).expand(-1, y_.shape[1], -1), 0)

        x_ = F.pad(x_, (pad_size, pad_size), "constant", 0)
        y_ = F.pad(y_, (pad_size, pad_size), "constant", 0)

        x_, y_ = self.to_YUV(x_, y_, dataL)

        assert torch.isfinite(x_).all(), "x is not finite"
        assert torch.isfinite(y_).all(), "y is not finite"

        return x_, y_

    def postprocessing(self, batch):
        x, y, dataL = batch
        x_, y_ = x, y

        x_, y_ = self.to_XYZ(x_, y_, dataL)  #

        assert torch.isfinite(x_).all(), "x is not finite"
        assert torch.isfinite(y_).all(), "y is not finite"

        return x_, y_

    def find_pad_size(self, x):
        pad_size = 0
        original_size = x.size(-1)
        power = np.ceil(np.log2(original_size))
        new_size = 2**power
        pad_size = int((new_size - original_size) // 2)

        return 0

    def mask_generation(self, dataL, length, batch_size):
        mask = torch.arange(length).expand(batch_size, length).to(dataL.device)
        mask = mask < dataL.unsqueeze(1)
        mask = torch.logical_not(mask)
        return mask

    def sample_noise(self, batch, pad_size, dataL, do_noise=True):
        result = self.transform(batch) if do_noise else batch
        x, y = result
        x, y = x.clone().detach(), y.clone().detach()

        x, y = self.preprocessing((x, y, dataL))

        assert torch.isfinite(x).all(), "x is not finite"
        assert torch.isfinite(y).all(), "y is not finite"

        return x, y

    def _generate_metrics(self, suffix):
        metrics = {
            f"metric_acc_pearson_X/{suffix}": PearsonMetric_acc_X(),
            f"metric_acc_pearson_Y/{suffix}": PearsonMetric_acc_Y(),
            f"metric_acc_pearson_Z/{suffix}": PearsonMetric_acc_Z(),
            f"metric_acc_pearson_norm/{suffix}": PearsonMetric_acc_Norm(),
            f"metric_acc_simVector/{suffix}": CosSimMetric_acc(),
            f"metric_acc_simVector_X/{suffix}": CosSimMetric_acc_X(),
            f"metric_acc_simVector_Y/{suffix}": CosSimMetric_acc_Y(),
            f"metric_acc_simVector_Z/{suffix}": CosSimMetric_acc_Z(),
            f"metric_acc_simVector_norm/{suffix}": CosSimMetric_acc_Norm(),
            f"metric_naive_distance_error_X/{suffix}": NaiveDistanceError_X(),
            f"metric_naive_distance_error_Y/{suffix}": NaiveDistanceError_Y(),
            f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError_Z(),
            f"metric_naive_distance_error_XY/{suffix}": NaiveDistanceError_XY(),
            f"metric_naive_distance_error/{suffix}": NaiveDistanceError(),
            f"metric_gyr_pearson_X/{suffix}": PearsonMetric_gyr_X(),
            f"metric_gyr_pearson_Y/{suffix}": PearsonMetric_gyr_Y(),
            f"metric_gyr_pearson_Z/{suffix}": PearsonMetric_gyr_Z(),
            f"metric_gyr_pearson_norm/{suffix}": PearsonMetric_gyr_Norm(),
            f"metric_gyr_simVector/{suffix}": CosSimMetric_gyr(),
            f"metric_gyr_simVector_X/{suffix}": CosSimMetric_gyr_X(),
            f"metric_gyr_simVector_Y/{suffix}": CosSimMetric_gyr_Y(),
            f"metric_gyr_simVector_Z/{suffix}": CosSimMetric_gyr_Z(),
            f"metric_gyr_simVector_norm/{suffix}": CosSimMetric_gyr_Norm(),
            f"metric_naive_Angular_error_X/{suffix}": NaiveAngularError_X(),
            f"metric_naive_Angular_error_Y/{suffix}": NaiveAngularError_Y(),
            f"metric_naive_Angular_error_Z/{suffix}": NaiveAngularError_Z(),
            f"metric_naive_Angular_error/{suffix}": NaiveAngularError(),
        }

        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics


class baseLineTestingPL(baseDiffusionModule):
    @torch.no_grad()
    def baseline_testing(self, batch):
        epsilon, x, dataL = batch
        estimate = epsilon.clone().detach()
        estimate[:, :3] = estimate[:, :3].cumsum(dim=-1) * 0.01
        x[:, :3] = x[:, :3].cumsum(dim=-1) * 0.01
        mask = self.mask_generation(dataL, x.shape[-1], x.shape[0])
        if torch.any(mask):
            estimate = estimate.masked_fill(
                mask.unsqueeze(1).expand(-1, estimate.shape[1], -1), 0
            )
            x = x.masked_fill(mask.unsqueeze(1).expand(-1, x.shape[1], -1), 0)

        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(estimate, x)

        self.log_dict(
            metrics,
            sync_dist=True,
        )

        return estimate

    def test_step(self, batch, batch_idx):
        estimate = self.baseline_testing(batch=batch)

        return torch.nan
