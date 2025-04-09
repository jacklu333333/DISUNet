import torch
import torch.nn as nn
import numpy as np


class mModule(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        for k, v in config.items():
            setattr(self, k, v)


class rotationNoise(mModule):
    def __init__(self, config):
        super(rotationNoise, self).__init__(config)

    def genRotationMatrix(self, yaw, pitch, roll):
        Rz = torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), 0],
                [torch.sin(yaw), torch.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        Ry = torch.tensor(
            [
                [torch.cos(pitch), 0, torch.sin(pitch)],
                [0, 1, 0],
                [-torch.sin(pitch), 0, torch.cos(pitch)],
            ]
        )
        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(roll), -torch.sin(roll)],
                [0, torch.sin(roll), torch.cos(roll)],
            ]
        )
        return Rz @ Ry @ Rx

    def forward(self, data):
        p = torch.rand(1).item()
        if p > self.probability:
            return data

        x, y = data
        acc = x[:3]
        gyr = x[3:6]

        if self.mode == "random":
            yaw = torch.rand(1) * 2 * torch.pi
            pitch = torch.rand(1) * 2 * torch.pi
            roll = torch.rand(1) * 2 * torch.pi
        elif self.mode == "axis":
            yaw = torch.randint(0, 4, (1,)) * torch.pi / 2
            pitch = torch.randint(0, 4, (1,)) * torch.pi / 2
            roll = torch.randint(0, 4, (1,)) * torch.pi / 2
        elif self.mode == "XY":
            yaw = torch.rand(1) * 2 * torch.pi
            pitch = torch.zeros(1)
            roll = torch.zeros(1)
        elif self.mode == "wabble":
            mean = torch.zeros(1)
            std = torch.ones(1) / 3
            yaw = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
            pitch = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
            roll = self.degree / 180 * torch.normal(mean=mean, std=std) * torch.pi / 2
        else:
            raise ValueError("mode should be random, axis, XY, wabble")

        R = self.genRotationMatrix(yaw, pitch, roll).to(x.device)

        acc = R @ acc
        gyr = R @ gyr
        if self.label_transform:
            newVel = y.clone().detach()
            for i in range(3, y.shape[0] + 1, 3):
                newVel[i - 3 : i] = R @ y[i - 3 : i]
        else:
            newVel = y.clone()

        return torch.cat([acc, gyr], dim=0), newVel

class gaussianNoise(mModule):
    def __init__(self, config):
        super(gaussianNoise, self).__init__(config)

    def forward(self, data):
        p = torch.rand(1).item()
        if p > self.probability:
            return data

        x, y = data
        acc = x[:3]
        gyr = x[3:6]

        acc = torch.normal(mean=acc, std=self.accNoise)
        gyr = torch.normal(mean=gyr, std=self.gyrNoise)

        return torch.cat([acc, gyr], dim=0), y


class scaleNoise(mModule):
    def __init__(self, config):
        super(scaleNoise, self).__init__(config)

    def forward(self, data):
        p = torch.rand(1).item()
        if p > self.probability:
            return data

        x, y = data
        acc = x[:3]
        gyr = x[3:6]

        acc *= (
            torch.normal(
                mean=torch.tensor([1.0] * 3), std=torch.tensor([self.accNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )

        gyr *= (
            torch.normal(
                mean=torch.tensor([1.0] * 3), std=torch.tensor([self.gyrNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )

        return torch.cat([acc, gyr], dim=0), y


class shiftNoise(mModule):
    def __init__(self, config):
        super(shiftNoise, self).__init__(config)

    def forward(self, data):
        p = torch.rand(1).item()
        if p > self.probability:
            return data

        x, y = data
        acc = x[:3]
        gyr = x[3:6]

        acc += (
            torch.normal(
                mean=torch.tensor([0.0] * 3), std=torch.tensor([self.accNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )
        gyr += (
            torch.normal(
                mean=torch.tensor([0.0] * 3), std=torch.tensor([self.gyrNoise / 3] * 3)
            )
            .reshape(3, 1)
            .to(x.device)
        )
        return torch.cat([acc, gyr], dim=0), y


class axisMasking(mModule):
    def __init__(self, config):
        super(axisMasking, self).__init__(config)

    def forward(self, data):
        p = torch.rand(1).item()
        if p > self.probability:
            return data

        x, y = data
        acc = x[:3].clone()
        gyr = x[3:6].clone()
        mag = x[6:].clone()

        mask_number = torch.randint(0, self.max_channel + 1, (1,)).item()
        mask = torch.randperm(3)[:mask_number]
        acc[mask] = 0
        gyr[mask] = 0
        mag[mask] = 0

        newVel = y.clone()
        newVel[mask] = 0

        return torch.cat([acc, gyr, mag], dim=0), newVel


class speedMasking(mModule):
    def __init__(self, config):
        super(speedMasking, self).__init__(config)

    def masker(self, y):
        mask = torch.where(torch.norm(y, dim=0) > self.config["threshold"])
        if len(mask[0]) == 0:
            mask = y.shape[1]
        else:
            mask = mask[0][0]

        maskVel = y.clone()
        maskVel[:, :mask] = 0

        return maskVel

    def forward(self, data):
        p = torch.rand(1).item()
        if p > self.probability:
            return data

        x, y = data
        newX = x.clone()
        newX[6:] = self.masker(newX[6:])
        newY = self.masker(y)

        return newX, newY


class keepSensorData(mModule):
    def __init__(self, config):
        super(keepSensorData, self).__init__(config)
        self.keepAcc = "acc" in self.config["keepSensor"]
        self.keepGyr = "gyr" in self.config["keepSensor"]
        self.keepMag = "Mag" in self.config["keepSensor"]

    def forward(self, data):
        x, y = data
        acc = x[0][:3]
        gyr = x[0][3:6]
        mag = x[0][6:]

        if not self.keepAcc:
            acc = torch.zeros_like(acc)
        if not self.keepGyr:
            gyr = torch.zeros_like(gyr)
        if not self.keepMag:
            mag = torch.zeros_like(mag)

        newX = torch.cat([acc, gyr, mag], dim=0)

        return newX, y


class trnasformBatch(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.transform = nn.Sequential(
            scaleNoise(self.config["scaleNoise"]),
            shiftNoise(self.config["shiftNoise"]),
            gaussianNoise(self.config["gaussianNoise"]),
            rotationNoise(self.config["rotationNoise"]),
        )

    def forward(self, batch):

        x, y = batch
        result_x = x.clone()
        result_y = y.clone()

        for i in range(x.shape[0]):
            result_x[i], result_y[i] = self.transform((result_x[i], result_y[i]))

        return (result_x, result_y)
