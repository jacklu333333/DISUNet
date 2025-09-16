import colored as cl
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torchmetrics import (
    KLDivergence,
    LogCoshError,
    PearsonCorrCoef,
    RelativeSquaredError,
)


class PearsonMetric(nn.Module):
    def __init__(self):
        super(PearsonMetric, self).__init__()
        self.esp = 1e-8

    def forward(self, x, y):
        x_temp = x.sum(dim=-1)
        y_temp = y.sum(dim=-1)

        x_temp = rearrange(x_temp, "b c -> c b")
        y_temp = rearrange(y_temp, "b c -> c b")

        vx = x_temp - x_temp.mean(dim=-1, keepdim=True)
        vy = y_temp - y_temp.mean(dim=-1, keepdim=True)

        if (vx == 0).all():
            print(cl.Fore.yellow, "all the vx are zeros", cl.Style.reset)
        if (vx == 0).all():
            print(cl.Fore.yellow, "all the vy are zeros", cl.Style.reset)
        if (vx == 0).all() or (vx == 0).all():
            print(cl.Fore.yellow, "-" * 200, cl.Style.reset)
        if not torch.isfinite(x).all():
            print(cl.Fore.yellow, "x is not finite", cl.Style.reset)
        if not torch.isfinite(y).all():
            print(cl.Fore.yellow, "y is not finite", cl.Style.reset)

        index = vx == 0
        vx[index] = vx[index] + self.esp

        index = vy == 0
        vy[index] = vy[index] + self.esp

        vx_root_square_sum = torch.sum(vx**2, dim=-1, keepdim=True).sqrt()
        vy_root_square_sum = torch.sum(vy**2, dim=-1, keepdim=True).sqrt()

        assert (vx_root_square_sum != 0).all(), f"{vx}"
        assert (vy_root_square_sum != 0).all(), f"{vy}"

        assert torch.isfinite(vx_root_square_sum).all(), f"{vx}"
        assert torch.isfinite(vy_root_square_sum).all(), f"{vy}"

        loss = torch.sum(vx * vy, dim=-1, keepdim=True) / (
            vx_root_square_sum * vy_root_square_sum
        )

        assert torch.isfinite(loss).all()
        return {
            "mean": loss.mean(),
        }


class PearsonMetric_acc_X(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_acc_X, self).__init__()

    def forward(self, x, y):
        tempX = x[:, 0:1]
        tempY = y[:, 0:1]
        result = super(PearsonMetric_acc_X, self).forward(tempX, tempY)

        return result


class PearsonMetric_acc_Y(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_acc_Y, self).__init__()

    def forward(self, x, y):
        tempX = x[:, 1:2]
        tempY = y[:, 1:2]
        result = super(PearsonMetric_acc_Y, self).forward(tempX, tempY)

        return result


class PearsonMetric_acc_Z(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_acc_Z, self).__init__()

    def forward(self, x, y):
        tempX = x[:, 2:3]
        tempY = y[:, 2:3]
        result = super(PearsonMetric_acc_Z, self).forward(tempX, tempY)

        return result


class PearsonMetric_acc_Norm(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_acc_Norm, self).__init__()

    def forward(self, x, y):
        x_temp = x[:, :3].sum(dim=-1).norm(dim=-1, keepdim=True)
        y_temp = y[:, :3].sum(dim=-1).norm(dim=-1, keepdim=True)

        x_temp = rearrange(x_temp, "b c -> c b")  # (batch, c, l) --> (batch, c)
        y_temp = rearrange(y_temp, "b c -> c b")  # (batch, c, l) --> (batch, c)

        vx = x_temp - x_temp.mean(dim=-1, keepdim=True)
        vy = y_temp - y_temp.mean(dim=-1, keepdim=True)

        if (vx == 0).all():
            print(cl.Fore.yellow, "all the vx are zeros", cl.Style.reset)
        if (vx == 0).all():
            print(cl.Fore.yellow, "all the vy are zeros", cl.Style.reset)
        if (vx == 0).all() or (vx == 0).all():
            print(cl.Fore.yellow, "-" * 200, cl.Style.reset)
        if not torch.isfinite(x).all():
            print(cl.Fore.yellow, "x is not finite", cl.Style.reset)
        if not torch.isfinite(y).all():
            print(cl.Fore.yellow, "y is not finite", cl.Style.reset)

        index = vx == 0
        vx[index] = vx[index] + self.esp
        index = vy == 0
        vy[index] = vy[index] + self.esp

        vx_root_square_sum = torch.sum(vx**2, dim=-1, keepdim=True).sqrt()
        vy_root_square_sum = torch.sum(vy**2, dim=-1, keepdim=True).sqrt()

        assert (vx_root_square_sum != 0).all(), f"{vx}"
        assert (vy_root_square_sum != 0).all(), f"{vy}"

        assert torch.isfinite(vx_root_square_sum).all(), f"{vx}"
        assert torch.isfinite(vy_root_square_sum).all(), f"{vy}"

        loss = torch.sum(vx * vy, dim=-1, keepdim=True) / (
            vx_root_square_sum * vy_root_square_sum
        )

        assert torch.isfinite(loss).all()
        return {
            "mean": loss.mean(),
        }


class PearsonMetric_gyr_X(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_gyr_X, self).__init__()

    def forward(self, x, y):
        tempX = x[:, 3:4]
        tempY = y[:, 3:4]
        result = super(PearsonMetric_gyr_X, self).forward(tempX, tempY)

        return result


class PearsonMetric_gyr_Y(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_gyr_Y, self).__init__()

    def forward(self, x, y):
        tempX = x[:, 4:5]
        tempY = y[:, 4:5]
        result = super(PearsonMetric_gyr_Y, self).forward(tempX, tempY)

        return result


class PearsonMetric_gyr_Z(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_gyr_Z, self).__init__()

    def forward(self, x, y):
        tempX = x[:, 5:6]
        tempY = y[:, 5:6]
        result = super(PearsonMetric_gyr_Z, self).forward(tempX, tempY)

        return result


class PearsonMetric_gyr_Norm(PearsonMetric):
    def __init__(self):
        super(PearsonMetric_gyr_Norm, self).__init__()

    def forward(self, x, y):
        x_temp = x[:, 3:].sum(dim=-1).norm(dim=-1, keepdim=True)
        y_temp = y[:, 3:].sum(dim=-1).norm(dim=-1, keepdim=True)

        x_temp = rearrange(x_temp, "b c -> c b")
        y_temp = rearrange(y_temp, "b c -> c b")

        vx = x_temp - x_temp.mean(dim=-1, keepdim=True)
        vy = y_temp - y_temp.mean(dim=-1, keepdim=True)

        if (vx == 0).all():
            print(cl.Fore.yellow, "all the vx are zeros", cl.Style.reset)
        if (vx == 0).all():
            print(cl.Fore.yellow, "all the vy are zeros", cl.Style.reset)
        if (vx == 0).all() or (vx == 0).all():
            print(cl.Fore.yellow, "-" * 200, cl.Style.reset)
        if not torch.isfinite(x).all():
            print(cl.Fore.yellow, "x is not finite", cl.Style.reset)
        if not torch.isfinite(y).all():
            print(cl.Fore.yellow, "y is not finite", cl.Style.reset)

        index = vx == 0
        vx[index] = vx[index] + self.esp
        index = vy == 0
        vy[index] = vy[index] + self.esp

        vx_root_square_sum = torch.sum(vx**2, dim=-1, keepdim=True).sqrt()
        vy_root_square_sum = torch.sum(vy**2, dim=-1, keepdim=True).sqrt()

        assert (vx_root_square_sum != 0).all(), f"{vx}"
        assert (vy_root_square_sum != 0).all(), f"{vy}"

        assert torch.isfinite(vx_root_square_sum).all(), f"{vx}"
        assert torch.isfinite(vy_root_square_sum).all(), f"{vy}"

        loss = torch.sum(vx * vy, dim=-1, keepdim=True) / (
            vx_root_square_sum * vy_root_square_sum
        )

        assert torch.isfinite(loss).all()
        return {
            "mean": loss.mean(),
        }


class CosSimMetric(nn.Module):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric, self).__init__()
        self.loss = nn.CosineSimilarity(dim=dim)

    def forward(self, x, y):
        loss = self.loss(x, y)
        return {
            "mean": loss.mean(),
            "std": loss.std(),
        }


class CosSimMetric_acc_X(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_acc_X, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 0:1], y[:, 0:1])
        # return loss.mean()
        return super(CosSimMetric_acc_X, self).forward(x[:, 0:1], y[:, 0:1])


class CosSimMetric_acc_Y(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_acc_Y, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 1:2], y[:, 1:2])
        # return loss.mean()
        return super(CosSimMetric_acc_Y, self).forward(x[:, 1:2], y[:, 1:2])


class CosSimMetric_acc_Z(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_acc_Z, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 2:3], y[:, 2:3])
        # return loss.mean()
        return super(CosSimMetric_acc_Z, self).forward(x[:, 2:3], y[:, 2:3])


class CosSimMetric_acc_Norm(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_acc_Norm, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(
        #     x[:, :3].norm(dim=1, keepdim=True), y[:, :3].norm(dim=1, keepdim=True)
        # )
        # return loss.mean()
        return super(CosSimMetric_acc_Norm, self).forward(
            x[:, :3].norm(dim=1, keepdim=True), y[:, :3].norm(dim=1, keepdim=True)
        )


class CosSimMetric_acc(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_acc, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, :3], y[:, :3])
        # return loss.mean()
        return super(CosSimMetric_acc, self).forward(x[:, :3], y[:, :3])


class CosSimMetric_gyr_X(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_gyr_X, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 3:4], y[:, 3:4])
        # return loss.mean()
        return super(CosSimMetric_gyr_X, self).forward(x[:, 3:4], y[:, 3:4])


class CosSimMetric_gyr_Y(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_gyr_Y, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 4:5], y[:, 4:5])
        # return loss.mean()
        return super(CosSimMetric_gyr_Y, self).forward(x[:, 4:5], y[:, 4:5])


class CosSimMetric_gyr_Z(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_gyr_Z, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 5:6], y[:, 5:6])
        # return loss.mean()
        return super(CosSimMetric_gyr_Z, self).forward(x[:, 5:6], y[:, 5:6])


class CosSimMetric_gyr_Norm(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_gyr_Norm, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(
        #     x[:, 3:].norm(dim=1, keepdim=True), y[:, 3:].norm(dim=1, keepdim=True)
        # )
        # return loss.mean()
        return super(CosSimMetric_gyr_Norm, self).forward(
            x[:, 3:].norm(dim=1, keepdim=True), y[:, 3:].norm(dim=1, keepdim=True)
        )


class CosSimMetric_gyr(CosSimMetric):
    def __init__(self, margin=0.1, dim=2):
        super(CosSimMetric_gyr, self).__init__(margin, dim)

    def forward(self, x, y):
        # loss = self.loss(x[:, 3:], y[:, 3:])
        # return loss.mean()
        return super(CosSimMetric_gyr, self).forward(x[:, 3:], y[:, 3:])


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()
        self.metric_x = PearsonMetric_acc_X()
        self.metric_y = PearsonMetric_acc_Y()
        self.metric_z = PearsonMetric_acc_Z()
        self.metric_norm = PearsonMetric_acc_Norm()

    def forward(self, x, y):
        loss_x = 1 - self.metric_x(x, y)["mean"]
        loss_y = 1 - self.metric_y(x, y)["mean"]
        loss_z = 1 - self.metric_z(x, y)["mean"]
        loss = loss_x + loss_y + loss_z
        return loss


class simclr_loss(nn.Module):
    def __init__(self):
        super(simclr_loss, self).__init__()
        self.metric_x = CosSimMetric_acc_X()
        self.metric_y = CosSimMetric_acc_Y()
        self.metric_z = CosSimMetric_acc_Z()
        self.metric_norm = CosSimMetric_acc_Norm()

    def forward(self, x, y):
        loss_x = 1 - self.metric_x(x, y)["mean"]
        loss_y = 1 - self.metric_y(x, y)["mean"]
        loss_z = 1 - self.metric_z(x, y)["mean"]
        loss = loss_x + loss_y + loss_z
        return loss


class KL_div_X(nn.Module):
    def __init__(self):
        super(KL_div_X, self).__init__()
        self.metric_x = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_x(x[:, 0], y[:, 0])
        return loss


class KL_div_Y(nn.Module):
    def __init__(self):
        super(KL_div_Y, self).__init__()
        self.metric_y = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_y(x[:, 1], y[:, 1])
        return loss


class KL_div_Z(nn.Module):
    def __init__(self):
        super(KL_div_Z, self).__init__()
        self.metric_z = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_z(x[:, 2], y[:, 2])
        return loss


class KL_div_Norm(nn.Module):
    def __init__(self):
        super(KL_div_Norm, self).__init__()
        self.metric_norm = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_norm(x.norm(dim=1), y.norm(dim=1))
        return loss


class KL_div(nn.Module):
    def __init__(self):
        super(KL_div, self).__init__()
        self.kl_div_x = KL_div_X()
        self.kl_div_y = KL_div_Y()
        self.kl_div_z = KL_div_Z()

    def forward(self, x, y):
        loss_x = self.kl_div_x(x, y)
        loss_y = self.kl_div_y(x, y)
        loss_z = self.kl_div_z(x, y)
        loss = loss_x + loss_y + loss_z
        return loss


class NaiveDistanceError(nn.Module):
    def __init__(self, rescale=1):
        super(NaiveDistanceError, self).__init__()
        self.rescale = rescale

    def forward(self, x, y):
        loss = (
            0.5
            * (x.cumsum(dim=-1).sum(dim=-1) - y.cumsum(dim=-1).sum(dim=-1)).norm(dim=-1)
            * 0.01**2
        ) / 5.12
        return {
            "mean": loss.mean(),
            "std": loss.std(),
        }


class NaiveDistanceError_X(NaiveDistanceError):
    def __init__(self):
        super(NaiveDistanceError_X, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveDistanceError_X, self).forward(x[:, 0:1], y[:, 0:1])
        return loss


class NaiveDistanceError_Y(NaiveDistanceError):
    def __init__(self):
        super(NaiveDistanceError_Y, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveDistanceError_Y, self).forward(x[:, 1:2], y[:, 1:2])
        return loss


class NaiveDistanceError_XY(NaiveDistanceError):
    def __init__(self):
        super(NaiveDistanceError_XY, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveDistanceError_XY, self).forward(x[:, 0:2], y[:, 0:2])
        return loss


class NaiveDistanceError_Z(NaiveDistanceError):
    def __init__(self):
        super(NaiveDistanceError_Z, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveDistanceError_Z, self).forward(x[:, 2:3], y[:, 2:3])
        return loss


class NaiveAngularError(nn.Module):
    def __init__(self, rescale=1):
        super(NaiveAngularError, self).__init__()
        self.rescale = rescale

    def forward(self, x, y):
        loss = (x.sum(dim=-1) - y.sum(dim=-1)).norm(dim=-1) * 0.01 / 5.12
        return {
            "mean": loss.mean(),
            "std": loss.std(),
        }


class NaiveAngularError_X(NaiveAngularError):
    def __init__(self):
        super(NaiveAngularError_X, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveAngularError_X, self).forward(x[:, 3:4], y[:, 3:4])
        return loss


class NaiveAngularError_Y(NaiveAngularError):
    def __init__(self):
        super(NaiveAngularError_Y, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveAngularError_Y, self).forward(x[:, 4:5], y[:, 4:5])
        return loss


class NaiveAngularError_Z(NaiveAngularError):
    def __init__(self):
        super(NaiveAngularError_Z, self).__init__()

    def forward(self, x, y):
        loss = super(NaiveAngularError_Z, self).forward(x[:, 5:6], y[:, 5:6])
        return loss


class CauchyLoss(nn.Module):
    def __init__(self, rescale=0.01):
        super(CauchyLoss, self).__init__()
        self.rescale = rescale

    def forward(self, x, y):
        loss = (x.sum(dim=-1) - y.sum(dim=-1)).square() * self.rescale
        return loss.mean()


class TukeyBiweightLoss(nn.Module):
    def __init__(self, rescale=0.01):
        super(TukeyBiweightLoss, self).__init__()
        self.rescale = rescale
        raise NotImplementedError

    def forward(self, x, y):
        diff = (x - y).abs() / self.rescale


class FairLoss(nn.Module):
    def __init__(self, rescale=0.01):
        super(FairLoss, self).__init__()
        self.rescale = rescale

    def forward(self, x, y):
        diff = (x - y).abs() / self.rescale
        index = diff < 1
        loss = torch.zeros_like(diff)
        loss[index] = 1 - diff[index].square()
        loss[~index] = 2 * diff[~index] - 1
        return loss.mean()


class myspecialLoss(nn.Module):
    def __init__(self, config):
        super(myspecialLoss, self).__init__()
        self.config = config
        self.history = {
            "train": [],
            "val": [],
            "test": [],
        }

        self.hubber_loss = nn.HuberLoss(delta=1)
        self.mse_loss = nn.MSELoss()
        self.pearson_loss = PearsonLoss()
        self.simclr = simclr_loss()

        self.naive_distance_error = NaiveDistanceError()
        self.naive_distance_error_X = NaiveDistanceError_X()
        self.naive_distance_error_Y = NaiveDistanceError_Y()
        self.naive_distance_error_Z = NaiveDistanceError_Z()
        self.accerlation_loss = accelerationError()
        self.velocity_loss = velocityError()
        self.position_loss = positionError()
        self.restoration_regulization = nn.MSELoss()

    def clear_history(self, mode="train"):
        self.history[mode].clear()

    def forward(self, x, y, restoration, l1, l2, pad_size, aux_loss, mode) -> dict:
        l1_out = l1
        l2_out = l2

        hubber_loss = self.hubber_loss(x, y)
        mse_loss = self.mse_loss(x * 100, y * 100)

        pearson_loss = self.pearson_loss(
            x,
            y,
        )
        simclr_loss = self.simclr(
            x,
            y,
        )
        naive_distance_error = self.naive_distance_error(x, y)["mean"]
        navie_distance_error_X = self.naive_distance_error_X(x, y)["mean"]
        navie_distance_error_Y = self.naive_distance_error_Y(x, y)["mean"]
        navie_distance_error_Z = self.naive_distance_error_Z(x, y)["mean"]

        accelerationLoss = self.accerlation_loss(x, y)
        velocityLoss = self.velocity_loss(x, y)
        positionLoss = self.position_loss(x, y)

        restoration_regularization = restoration.square().mean()

        if len(self.history[mode]) == 0:
            self.history[mode].append(
                {
                    "loss_hubber": hubber_loss.clone().detach(),
                    "loss_mse": mse_loss.clone().detach(),
                    "loss_pearson": pearson_loss.clone().detach(),
                    "loss_simclr": simclr_loss.clone().detach(),
                    "loss_l1": l1_out.clone().detach(),
                    "loss_l2": l2_out.clone().detach(),
                    "loss_aux": aux_loss.clone().detach(),
                    "naive_distance_error": naive_distance_error.clone().detach(),
                    "naive_distance_error_X": navie_distance_error_X.clone().detach(),
                    "naive_distance_error_Y": navie_distance_error_Y.clone().detach(),
                    "naive_distance_error_Z": navie_distance_error_Z.clone().detach(),
                    # "loss_distance_cumsum": distance_cumsum_loss.clone().detach(),
                    "loss_acceleration": accelerationLoss.clone().detach(),
                    "loss_velocity": velocityLoss.clone().detach(),
                    "loss_position": positionLoss.clone().detach(),
                    "restoration_regularization": restoration_regularization.clone().detach(),
                }
            )
        l1_out = l1_out / self.history[mode][0]["loss_l1"]
        l2_out = l2_out / self.history[mode][0]["loss_l2"]
        aux_loss_ = aux_loss / self.history[mode][0]["loss_aux"]
        restoration_regularization = (
            restoration_regularization
            / self.history[mode][0]["restoration_regularization"]
        )

        losses = {
            f"loss_hubber/{mode}": hubber_loss,
            f"loss_mse/{mode}": mse_loss,
            f"loss_pearson/{mode}": pearson_loss,
            f"loss_simclr/{mode}": simclr_loss,
            f"loss_l1/{mode}": l1_out,
            f"loss_l2/{mode}": l2_out,
            f"loss_aux/{mode}": aux_loss_,
            f"naive_distance_error/{mode}": naive_distance_error,
            f"naive_distance_error_X/{mode}": navie_distance_error_X,
            f"naive_distance_error_Y/{mode}": navie_distance_error_Y,
            f"naive_distance_error_Z/{mode}": navie_distance_error_Z,
            # f"loss_distance_cumsum/{mode}": distance_cumsum_loss,
            f"loss_acceleration/{mode}": accelerationLoss,
            f"loss_velocity/{mode}": velocityLoss,
            f"loss_position/{mode}": positionLoss,
            f"restoration_regularization/{mode}": restoration_regularization,
        }
        total_loss = 0
        for key, value in losses.items():
            total_loss += value * self.config[key.split("/")[0]]
        total_loss = total_loss / sum(self.config.values())

        losses[f"loss/{mode}"] = total_loss
        return losses


class accelerationError(nn.Module):
    def __init__(self):
        super(accelerationError, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        loss = self.loss(x[:, :3], y[:, :3])
        return loss.mean()


class velocityError(nn.Module):
    def __init__(self):
        super(velocityError, self).__init__()

        self.loss = nn.L1Loss()

    def forward(self, x, y):
        loss = x[:, :3].cumsum(dim=-1) - y[:, :3].cumsum(dim=-1)
        loss = loss.norm(dim=1)
        loss = F.mse_loss(loss, torch.zeros_like(loss))
        return loss.mean()


class positionError(nn.Module):
    def __init__(self):
        super(positionError, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        loss = self.loss(
            x[:, :3].cumsum(dim=-1).cumsum(dim=-1) * 0.01**2,
            y[:, :3].cumsum(dim=-1).cumsum(dim=-1) * 0.01**2,
        )
        return loss.mean()


class angularVelError(nn.Module):
    def __init__(self):
        super(angularVelError, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        assert x.shape[1] == 6
        assert y.shape[1] == 6

        loss = x[:, 3:] - y[:, 3:]
        loss = loss.norm(dim=1)
        loss = F.mse_loss(loss, torch.zeros_like(loss))
        return loss.mean()


class angularError(nn.Module):
    def __init__(self):
        super(angularError, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        loss = self.loss(
            x[:, 3:].cumsum(dim=-1) * 0.01,
            y[:, 3:].cumsum(dim=-1) * 0.01,
        )
        return loss.mean()
