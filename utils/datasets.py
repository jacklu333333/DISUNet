import datetime
import glob
import json
import math
import os
import pickle
from pathlib import Path

import colored as cl
import cudf.pandas
import h5py
import numpy as np

# import cudf as pd
import pandas as pd

cudf.pandas.install()
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import scipy.signal as signal
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
from einops import rearrange
from einops.layers.torch import Rearrange
from overrides import overrides
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from scipy.spatial.transform import Rotation as R
from torch.nn.modules.container import Sequential
from tqdm import tqdm

from .correction import rotateToWorldFrame
from .geoPreprocessor import GravityRemoval, MagneticRemoval
from .stepDetection import find_steps
from .transform import *

SCALE = {
    # "RoNIN": {
    #     "acc": 60.0,
    #     "gyr": 25.0,
    #     "label": 60,
    # },
    "RoNIN": {
        "acc": 4.0,
        "gyr": 4.0,
        "label": 4.0,
    },
    # "OIOD": {
    #     "acc": 130.0,
    #     "gyr": 10.0,
    #     "label": 130.0,
    # },
    "OIOD": {
        "acc": 4.0,
        "gyr": 4.0,
        "label": 4.0,
    },
}


def peeking_trajectory(location):

    # plot 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(location[:, 0], location[:, 1], location[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # set tri-axis equal
    ax.set_box_aspect(
        [np.ptp(location[:, 0]), np.ptp(location[:, 1]), np.ptp(location[:, 2])]
    )
    plt.show()
    c = input("Continue? [y/n]")
    if c == "n":
        exit()
    plt.close("all")


class mDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        mode="train",
        transform: torch.nn.modules.container.Sequential = None,
        window_size=100,
        stride=10,
        keep_filters: list = None,
        skip_filters: list = None,
    ):
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.keep_filters = keep_filters
        self.skip_filters = skip_filters

    def __len__(self):
        assert len(self.dataX) == len(self.dataY)
        return len(self.dataX)

    def __getitem__(self, idx):
        if isinstance(self.dataX[idx], np.ndarray):
            x = torch.from_numpy(self.dataX[idx]).float()
            y = torch.from_numpy(self.dataY[idx]).float()
        else:
            x = self.dataX[idx].float().clonse().detach()
            y = self.dataY[idx].float().clonse().detach()

        if self.transform is not None:
            x, y = self.transform((x, y))

        return x[:6], y, self.dataL[idx]

    def get_path_format(self):
        return f"{self.mode}_{self.window_size}_{self.stride}_dataset"

    def check_existence(self):
        return Path(self.root).joinpath(self.get_path_format()).exists()

    def save(self):
        pass

    def load(self):
        path = Path(self.root).joinpath(self.get_path_format())
        print(cl.Fore.red + f"Shortcut" + cl.Style.reset)
        print(cl.Fore.yellow + f"Loading {path}" + cl.Style.reset)
        self.dataX = torch.load(
            path.joinpath("dataX.pth"), mmap=True, weights_only=False
        )
        self.dataY = torch.load(
            path.joinpath("dataY.pth"), mmap=True, weights_only=False
        )

    def split_data(self, acc, gyr, mag, velocity, acceleration, window_size, stride):
        assert (
            acc.shape[0] == gyr.shape[0] == mag.shape[0] == velocity.shape[0]
        ), f"{acc.shape} != {gyr.shape} != {mag.shape} != {velocity.shape}"
        zeros_front = torch.zeros((window_size, 3)).cuda()
        prevel = torch.cat([zeros_front, velocity], dim=0)

        # use window_size to split the data with stride
        accSplit = acc.unfold(0, window_size, stride)
        gyrSplit = gyr.unfold(0, window_size, stride)
        magSplit = mag.unfold(0, window_size, stride)
        prevel = prevel.unfold(0, window_size, stride)[: accSplit.shape[0]]
        label = acceleration.unfold(0, window_size, stride)
        assert (
            accSplit.shape[0] == gyrSplit.shape[0] == prevel.shape[0] == label.shape[0]
        )

        return accSplit, gyrSplit, magSplit, label

    def read_list(self):
        raise NotImplementedError

    def load_files(self):
        raise NotImplementedError

    def _load_single_file(self):
        raise NotImplementedError

    def vel_acc_generator(self, location, sample_rate=100.0):
        if isinstance(location, np.ndarray):
            location = torch.from_numpy(location).float().cuda()
        location = location - location[:1, :]
        velocity = (
            torch.diff(torch.cat([location[:1], location], dim=0), dim=0) * sample_rate
        )
        acceleration = (
            torch.diff(
                torch.cat([torch.zeros(1, 3).to(velocity), velocity], dim=0), dim=0
            )
            * sample_rate
        )

        assert (
            acceleration.shape == velocity.shape == location.shape
        ), f"{acceleration.shape} != {velocity.shape} != {location.shape}"
        return velocity, acceleration

    def _time_series_filter(self, window_size=5, *args):
        size = len(args[0])
        weights = torch.ones((window_size)) / window_size
        weights = weights.reshape(1, 1, window_size)
        result = []
        for arg in args:
            filter_data = arg
            # check tensor or numpy
            if isinstance(filter_data, np.ndarray):
                filter_data = torch.from_numpy(arg).float().cuda()
            from scipy.signal import savgol_filter

            filter_data = savgol_filter(
                arg.cpu().numpy(),
                window_length=window_size,
                axis=0,
                polyorder=1,
                mode="nearest",
            )
            filter_data = torch.from_numpy(filter_data).float().cuda()
            assert (
                arg.shape == filter_data.shape
            ), f"The shape is not the same after savgol filter {arg.shape} != {filter_data.shape}"

            if not isinstance(filter_data, torch.Tensor):
                filter_data = filter_data.cpu().numpy()
            filter_data = (
                torch.nn.functional.conv1d(
                    torch.nn.functional.pad(
                        filter_data.swapaxes(0, 1).unsqueeze(1),
                        (window_size // 2, window_size // 2),
                        mode="replicate",
                    ),
                    weights.to(filter_data.device),
                    stride=1,
                )
                .squeeze(1)
                .swapaxes(0, 1)
            )
            filter_data = filter_data[: arg.shape[0]]
            assert (
                arg.shape == filter_data.shape
            ), f"The shape is not the same after moving average filter {arg.shape} != {filter_data.shape}"
            result.append(filter_data)

        return result

    def _step_finder(self, acc_norm, window_size=5):
        if isinstance(acc_norm, torch.Tensor):
            acc_norm = acc_norm.cpu().numpy()
        smoothed_acc_norm = signal.savgol_filter(acc_norm, window_size, 2)

        _, valleys = find_steps(smoothed_acc_norm)

        valleys += (window_size - 1) // 2

        return valleys, smoothed_acc_norm

    def _first_motion(self, acc_norm, threshold=0.5, *args):
        if isinstance(acc_norm, np.ndarray):
            acc_norm = torch.from_numpy(acc_norm).float().cuda()
        diff = torch.diff(acc_norm)
        
        idx = torch.where(diff > threshold)[0][0]
        
        result = []
        for arg in args:
            result.append(arg[idx:])

        return result

    def split_by_step(
        self,
        stepIdx,
        acc,
        gyr,
        mag,
        velocity,
        acceleration,
        orientation,
        window_size,
        stride,
        modes="train",
    ):
        observation = torch.cat([acc, gyr, mag], dim=1)
        label = torch.cat([acceleration, orientation], dim=1)

        assert observation.shape[0] == label.shape[0]

        x = []
        y = []
        length = []

        if np.max(np.diff(stepIdx)) > window_size:
            print(
                cl.Fore.yellow,
                f"{np.max(np.diff(stepIdx))} > {window_size}",
                cl.Style.reset,
            )

        if modes != "test":
            for i in range(len(stepIdx)):
                for j in range(i + 1, len(stepIdx)):
                    if stepIdx[j] - stepIdx[i] < window_size:
                        continue
                    else:
                        j -= 1
                        o = observation[stepIdx[i] : stepIdx[j]]
                        l = label[stepIdx[i] : stepIdx[j]]
                        length.append(stepIdx[j] - stepIdx[i])
                        o, l = self.package_to_window_size(
                            stepIdx[j] - stepIdx[i], o, l, window_size
                        )
                        x.append(o)
                        y.append(l)
                        break
        else:
            prev = 0
            i = 1
            while i < len(stepIdx):
                if i + 1 < len(stepIdx):
                    if stepIdx[i + 1] - prev < window_size:
                        i = i + 1
                        continue
                    else:
                        o = observation[prev : stepIdx[i]]
                        l = label[prev : stepIdx[i]]
                        if len(o) > window_size:
                            prev = stepIdx[i]
                            continue

                        length.append(stepIdx[i] - prev)
                        o, l = self.package_to_window_size(
                            stepIdx[i] - prev, o, l, window_size
                        )
                        x.append(o)
                        y.append(l)
                        prev = stepIdx[i]
                else:
                    o = observation[prev : stepIdx[i]]
                    l = label[prev : stepIdx[i]]
                    length.append(stepIdx[i] - prev)
                    o, l = self.package_to_window_size(
                        stepIdx[i] - prev, o, l, window_size
                    )
                    x.append(o)
                    y.append(l)
                    prev = stepIdx[i]
                i += 1

        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        length = torch.tensor(length)
        assert (
            x.shape[-1] == y.shape[-1] == window_size
        ), f"{x.shape} != {y.shape} != {window_size}"
        return x, y, length

    def package_to_window_size(self, length, observation, label, window_size):
        assert length <= window_size, f"{length} > {window_size}"
        observation = observation.swapaxes(0, 1)
        label = label.swapaxes(0, 1)

        observation = F.pad(observation, (0, window_size - length), "constant", 0)
        label = F.pad(label, (0, window_size - length), "constant", 0)

        return observation, label

    def get_sequence(self, idx, mode="test"):
        """
        get the sequence of the data
        ----------------------------------------
        input:
            idx: int
                the index of the sequence
        ----------------------------------------
        output:
            x: torch.Tensor
                the input data
            y: torch.Tensor
                the output data
        """

        if isinstance(self, OIODDataset):
            files = self.files
            # remove
            if self.skip_filters is not None:
                if len(self.skip_filters) > 0:
                    files = [
                        (i, v)
                        for i, v in files
                        if all(key not in str(v) for key in self.skip_filters)
                    ]

            if self.keep_filters is not None:
                if len(self.keep_filters) > 0:
                    files = [
                        (i, v)
                        for i, v in files
                        if any(key in str(v) for key in self.keep_filters)
                    ]
            idx %= len(files)
            file = files[idx]
            imu, vi = file
            file = vi
            print(cl.Fore.yellow + f"Loading {imu} and {vi}" + cl.Style.reset)
            # print(cl.Fore.yellow + f"Loading {imu} and {vi}" + cl.Style.reset)
            x, y, length = self._load_single_file(
                imu, vi, window_size=self.window_size, stride=self.window_size
            )
        else:
            files = self.files
            file = files[idx]
            print(cl.Fore.yellow + f"Loading {file}" + cl.Style.reset)
            x, y, length = self._load_single_file(
                file, window_size=self.window_size, stride=self.window_size
            )
        x = torch.cat([x], dim=0)[:, :6]
        y = torch.cat([y], dim=0)[:, :6]
        length = torch.cat([length], dim=0)

        assert x.shape[0] == y.shape[0] == length.shape[0]
        dataset = torch.utils.data.TensorDataset(x, y, length)
        return dataset, file

    def _orientation2degpersec(self, orientation):
        length = orientation.shape[0]
        dt = 0.01
        # convert the orientation to radian per second
        orientation = R.from_quat(orientation)
        # compute the angular_velocity  base on the orientation
        angular_velocities = [np.zeros(3)]
        for i in range(1, length):
            # Compute the relative rotation from step i-1 to step i
            relative_rotation = orientation[i] * orientation[i - 1].inv()
            # Extract the angular velocity in the form of a rotation vector
            angular_velocity = relative_rotation.as_rotvec() / dt
            angular_velocities.append(angular_velocity)
        angular_velocities = np.array(angular_velocities)  # Convert list to numpy array
        angular_velocities = torch.tensor(angular_velocities).float().cuda()

        return angular_velocities


class RoNINDataset(mDataset):
    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        window_size=100,
        stride=10,
        keep_filters: list = None,
        skip_filters: list = None,
    ):
        super().__init__(
            root=root,
            mode=mode,
            transform=transform,
            window_size=window_size,
            stride=stride,
            keep_filters=keep_filters,
            skip_filters=skip_filters,
        )
        if mode != "test":
            self.files = self.read_list(self.root.joinpath(f"lists/list_{mode}.txt"))
        else:
            print(cl.Fore.yellow + "Test mode with [seen] data" + cl.Style.reset)
            self.files = self.read_list(
                self.root.joinpath(f"lists/list_{mode}_seen.txt")
            )

        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

        assert len(self.dataX) == len(
            self.dataY
        ), f"{len(self.dataX)} != {len(self.dataY)} with {self.dataX.shape} {self.dataY.shape}"

        print(cl.Fore.green + f"Loaded {len(self.dataY)} samples" + cl.Style.reset)

    def read_list(self, file):
        with open(file, "r") as f:
            files = f.readlines()
            files = [
                Path(self.root).joinpath("Data").joinpath(f.strip()) for f in files
            ]
        return files

    def load_files(self):
        self.dataX = []
        self.dataY = []
        self.dataL = []
        self.classes = []
        print(cl.Fore.yellow + f"Loading {self.mode} data" + cl.Style.reset)
        for file in tqdm(self.files):
            x, y, length, classes = self._load_single_file(
                file, window_size=self.window_size, stride=self.stride
            )
            self.dataX.append(x.detach().cpu().numpy())
            self.dataY.append(y.detach().cpu().numpy())
            self.dataL.append(length.detach().cpu().numpy())
            self.classes.append(classes.detach().cpu().numpy())
        self.dataX = np.concatenate(self.dataX, axis=0)
        self.dataY = np.concatenate(self.dataY, axis=0)
        self.dataL = np.concatenate(self.dataL, axis=0)
        self.classes = np.concatenate(self.classes, axis=0)

        assert (
            self.dataX.shape[0] == self.dataY.shape[0]
        ), f"{self.dataX.shape} != {self.dataY.shape}"
        assert self.dataX.shape[1] == 9, f"{self.dataX.shape[1]} != 9"
        assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        assert (
            self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        ), f"{self.window_size} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"

        self.save()

    def get_class(self, name, number):
        trajectory = [
            "a035",
            "a017",
            "a022",
            "a029",
            "a040",
            "a005",
            "a038",
            "a051",
            "a045",
            "a055",
            "a034",
            "a037",
            "a058",
            "a039",
            "a053",
            "a027",
            "a025",
            "a009",
            "a011",
            "a013",
            "a020",
            "a057",
            "a004",
            "a028",
            "a042",
            "a049",
            "a059",
            "a014",
            "a023",
            "a056",
            "a018",
            "a015",
            "a002",
            "a016",
            "a010",
            "a030",
            "a033",
            "a052",
            "a007",
            "a046",
            "a000",
            "a001",
            "a003",
            "a050",
            "a024",
            "a054",
            "a043",
            "a006",
            "a021",
            "a047",
            "a019",
            "a036",
            "a032",
            "a044",
            "a026",
            "a031",
            "a012",
        ]
        # print(len(trajectory))
        for idx, t in enumerate(trajectory):
            if t in name:
                return [idx for _ in range(number)]

        assert False, f"Class not found {name}"

    def _load_single_file(
        self, file: Path, window_size: int, stride: int, return_wave=False
    ):
        if not file.exists():
            print(cl.Fore.red + f"File {file} does not exist" + cl.Style.reset)

        # read the data from file
        f = h5py.File(file.joinpath("data.hdf5"), "r")
        json_info = json.load(open(file.joinpath("info.json")))
        date = json_info["date"]  # formate mm/dd/yy
        date = datetime.datetime.strptime(date, "%m/%d/%y")
        startFrame = int(math.ceil(json_info["start_frame"] / 2))
        dataX = []
        dataY = []
        dataL = []
        classes = []
        for i in range(2):
            # get the data from 200Hz to 100Hz
            acc = f["synced"]["linacce"][i::2, :]
            acc_norm = f["synced"]["acce"][i::2, :]
            gyr = f["synced"]["gyro"][i::2, :]
            mag = f["synced"]["magnet"][i::2, :] / 100
            location = f["pose"]["tango_pos"][i::2, :]
            orientation = f["pose"]["tango_ori"][i::2, :]

            orientation = R.from_quat(orientation).inv().as_quat()
            # # rotate to world frame
            acc, gyr, mag = rotateToWorldFrame(
                acc, gyr, mag, sample_rate=100.0, rotation=orientation
            )

            # convert to tensor on cuda and cut the data
            acc = torch.from_numpy(acc).float().cuda()[startFrame:]
            gyr = torch.from_numpy(gyr).float().cuda()[startFrame:]
            mag = torch.from_numpy(mag).float().cuda()[startFrame:]
            acc_norm = torch.from_numpy(acc_norm).float().cuda()[startFrame:]
            location = location[startFrame:]
            orientation = orientation[startFrame:]

            assert acc.shape[0] == gyr.shape[0] == mag.shape[0] == location.shape[0]

            gyr = gyr  # / np.pi
            mag = MagneticRemoval(mag=mag, location="RoNIN", rescale=True, date=date)[
                :, :3
            ]

            assert (
                acc.shape[0]
                == gyr.shape[0]
                == mag.shape[0]
                == location.shape[0]
                == orientation.shape[0]
            )

            acc_norm = torch.norm(acc, dim=1)
            assert acc_norm.shape[0] == acc.shape[0] == orientation.shape[0]
            acc, gyr, mag, location, acc_norm, orientation = self._first_motion(
                acc_norm,
                0.5,
                acc,
                gyr,
                mag,
                location,
                acc_norm,
                orientation,
            )

            orientation = self._orientation2degpersec(orientation)
            velocity, acceleration = self.vel_acc_generator(location, sample_rate=100.0)
            if return_wave:
                return acc, gyr, mag, velocity, acceleration

            valleys, acc_norm = self._step_finder(acc_norm, window_size=5)
            assert acc_norm.shape[0] == acc.shape[0] == orientation.shape[0]
            acc, gyr, mag, orientation, velocity, acceleration = (
                self._time_series_filter(
                    30, acc, gyr, mag, orientation, velocity, acceleration
                )
            )

            x, y, length = self.split_by_step(
                stepIdx=valleys,
                acc=acc,
                gyr=gyr,
                mag=mag,
                orientation=orientation,
                velocity=velocity,
                acceleration=acceleration,
                window_size=window_size,
                stride=stride,
                modes=self.mode,
            )

            assert (
                x.shape[0] == y.shape[0] == length.shape[0]
            ), f"{x.shape} != {y.shape}"
            assert x.shape[1] == 9, f"{x.shape[1]} != 9"

            dataX.append(x)
            dataY.append(y)
            dataL.append(length)
            classes.extend(self.get_class(str(file), x.shape[0]))
            if self.mode == "pred":
                break

        dataX = torch.cat(dataX, dim=0)
        dataY = torch.cat(dataY, dim=0)
        dataL = torch.cat(dataL, dim=0)
        classes = torch.tensor(classes).cuda()

        assert dataX.shape[0] == dataY.shape[0]
        assert dataX.shape[1] == 9, f"{file} has {dataX.shape[1]} columns"
        assert dataY.shape[1] == 6, f"{file} has {dataY.shape[1]} columns"

        return dataX, dataY, dataL, classes


class OIODDataset(mDataset):
    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        window_size=100,
        stride=10,
        keep_filters: list = None,
        skip_filters: list = None,
    ) -> None:
        super().__init__(
            root=root,
            mode=mode,
            transform=transform,
            window_size=window_size,
            stride=stride,
            keep_filters=keep_filters,
            skip_filters=skip_filters,
        )
        self._IMU_HEADERS = [
            "Time",
            "attitude_roll(radians)",
            "attitude_pitch(radians)",
            "attitude_yaw(radians)",
            "rotation_rate_x(radians/s)",
            "rotation_rate_y(radians/s)",
            "rotation_rate_z(radians/s)",
            "gravity_x(G)",
            "gravity_y(G)",
            "gravity_z(G)",
            "user_acc_x(G)",
            "user_acc_y(G)",
            "user_acc_z(G)",
            "magnetic_field_x(microteslas)",
            "magnetic_field_y(microteslas)",
            "magnetic_field_z(microteslas)",
        ]

        self._VI_HEADERS = [
            "Time",
            "Header",
            "translation.x",
            "translation.y",
            "translation.z",
            "rotation.x",
            "rotation.y",
            "rotation.z",
            "rotation.w",
        ]

        self.files = self.read_list(
            self.root.joinpath("lists").joinpath(f"{mode}_list.txt")
        )
        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

        assert (
            self.dataX.shape[0] == self.dataY.shape[0]
        ), f"{self.dataX.shape} != {self.dataY.shape}"
        assert self.dataX.shape[1] == 9, f"{self.dataX.shape[1]} != 9"
        assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        assert (
            self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        ), f"{self.window_size} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"

    def read_list(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
            files = []
            for line in lines:
                imu, vi = line.strip().split(",")
                # check existence of file
                assert Path(self.root).joinpath(imu).exists(), (
                    cl.Fore.red
                    + f"File {Path(self.root).joinpath(imu)} does not exist"
                    + cl.Style.reset
                )
                assert Path(self.root).joinpath(vi).exists(), (
                    cl.Fore.red
                    + f"File {Path(self.root).joinpath(vi)} does not exist"
                    + cl.Style.reset
                )
                files.append((imu, vi))
        print(
            cl.Fore.green
            + f"Found {len(files)} files for {self.mode}."
            + cl.Style.reset
        )
        return files

    def load_files(self):
        self.dataX = []
        self.dataY = []
        self.dataL = []
        self.classes = []
        for imu, vi in tqdm(self.files):
            if self.skip_filters is not None:
                if any([f in str(vi) for f in self.skip_filters]):
                    continue

            if self.keep_filters is not None:
                if not any([f in str(vi) for f in self.keep_filters]):
                    continue

            x, y, length, classes = self._load_single_file(
                imu, vi, window_size=self.window_size, stride=self.stride
            )
            self.dataX.append(x.detach().cpu().numpy())
            self.dataY.append(y.detach().cpu().numpy())
            self.dataL.append(length.detach().cpu().numpy())

        self.dataX = np.concatenate(self.dataX, axis=0)
        self.dataY = np.concatenate(self.dataY, axis=0)
        self.dataL = np.concatenate(self.dataL, axis=0)


        assert (
            self.dataX.shape[0] == self.dataY.shape[0] == self.dataL.shape[0]
        ), f"{self.dataX.shape} != {self.dataY.shape}"
        assert self.dataX.shape[1] == 9, f"{self.dataX.shape[1]} != 9"
        assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        assert (
            self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        ), f"{self.window_size} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"

        self.save()

    def get_class(self, name, number):
        activity = [
            "handbag",
            "handheld",
            "large scale",
            "multi devices",
            "multi users",
            "pocket",
            "running",
            "slow walking",
            "trolley",
        ]
        for idx, act in enumerate(activity):
            if act in name:
                return [idx for _ in range(number)]
        assert "label not found"

    def _load_single_file(
        self,
        imu,
        vi,
        window_size: int,
        stride: int,
        return_wave=False,
    ):
        df_imu = pd.read_csv(Path(self.root).joinpath(imu), names=self._IMU_HEADERS)
        df_vi = pd.read_csv(Path(self.root).joinpath(vi), names=self._VI_HEADERS)

        df_imu = df_imu.ffill()
        df_vi = df_vi.ffill()

        minlen = min(len(df_imu), len(df_vi))
        df_imu = df_imu[:minlen]
        df_vi = df_vi[:minlen]

        if "nexus" in str(vi):
            acc = df_imu[
                [
                    "gravity_x(G)",
                    "gravity_y(G)",
                    "gravity_z(G)",
                ]
            ].to_numpy()

        else:
            acc = df_imu[
                [
                    "user_acc_x(G)",
                    "user_acc_y(G)",
                    "user_acc_z(G)",
                ]
            ].to_numpy()

        gyr = df_imu[
            [
                "rotation_rate_x(radians/s)",
                "rotation_rate_y(radians/s)",
                "rotation_rate_z(radians/s)",
            ]
        ].to_numpy()
        mag = df_imu[
            [
                "magnetic_field_x(microteslas)",
                "magnetic_field_y(microteslas)",
                "magnetic_field_z(microteslas)",
            ]
        ].to_numpy()
        location = df_vi[["translation.x", "translation.y", "translation.z"]].to_numpy()
        rotation = df_vi[
            ["rotation.x", "rotation.y", "rotation.z", "rotation.w"]
        ].to_numpy()

        acc = acc * 9.81
        gyr = gyr
        mag = mag / 100 

        acc, gyr, mag = rotateToWorldFrame(acc, gyr, mag, rotation=rotation)

        if "nexus" in str(vi):
            acc_norm = np.linalg.norm(acc, axis=1)
            acc = GravityRemoval(acc=acc, location="Oxford", rescale=False)

        acc_norm = np.linalg.norm(acc, axis=1)
        acc, gyr, mag, location, acc_norm, rotation = self._first_motion(
            acc_norm,
            0.5,
            acc,
            gyr,
            mag,
            location,
            acc_norm,
            rotation,
        )
        valleys, smoothed_acc_norm = self._step_finder(acc_norm, window_size=5)

        acc = torch.from_numpy(acc).float().cuda()
        gyr = torch.from_numpy(gyr).float().cuda()
        mag = torch.from_numpy(mag).float().cuda()

        velocity, acceleration = self.vel_acc_generator(location, sample_rate=100.0)
        rotation = self._orientation2degpersec(rotation)

        assert (
            acceleration.shape == velocity.shape == location.shape == rotation.shape
        ), f"{acceleration.shape} != {velocity.shape} != {location.shape} != {rotation.shape}"
        del location

        assert acc.shape[0] == gyr.shape[0] == velocity.shape[0]
        assert acc.isfinite().all(), f"{imu} has NaN in acc"
        assert gyr.isfinite().all(), f"{imu} has NaN in gyr"
        assert velocity.isfinite().all(), f"{vi} has NaN in vel"

        acc, gyr, mag, rotation, velocity, acceleration = self._time_series_filter(
            30, acc, gyr, mag, rotation, velocity, acceleration
        )
        x, y, length = self.split_by_step(
            stepIdx=valleys,
            acc=acc,
            gyr=gyr,
            mag=mag,
            orientation=rotation,
            velocity=velocity,
            acceleration=acceleration,
            window_size=window_size,
            stride=stride,
            modes=self.mode,
        )
        label = self.get_class(vi, len(x))

        return x, y, length, label


class RIDIDataset(mDataset):
    def __init__(
        self,
        root,
        mode: str = "train",
        window_size: int = 100,
        stride: int = 1,
        transform: torch.nn.modules.container.Sequential = None,
        keep_filters: list = None,
        skip_filters: list = None,
    ) -> None:
        super().__init__(
            root=root,
            mode=mode,
            window_size=window_size,
            stride=stride,
            transform=transform,
            keep_filters=keep_filters,
            skip_filters=skip_filters,
        )

        self.files = self.read_list(self.root.joinpath(f"{mode}_list.txt"))
        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

        assert (
            self.dataX.shape[0] == self.dataY.shape[0]
        ), f"{self.dataX.shape} != {self.dataY.shape}"
        assert self.dataX.shape[1] == 9, f"{self.dataX.shape[1]} != 9"
        assert self.dataY.shape[1] == 6, f"{self.dataY.shape[1]} != 6"
        assert (
            self.window_size == self.dataX.shape[2] == self.dataY.shape[2]
        ), f"{self.config['window_size']} != {self.dataX.shape[2]} != {self.dataY.shape[2]}"
        print(cl.Fore.green + f"Loaded {len(self.dataY)} samples" + cl.Style.reset)

    def compute_statics(self):
        self.acc = []
        self.gyr = []
        self.l_acc = []
        self.l_gyr = []
        self.L = []

        lists = ["test", "train", "val"]
        files = []
        for mode in lists:
            files += self.read_list(self.root.joinpath(f"{mode}_list.txt"))

        for file in tqdm(files):
            x, y, length = self._load_single_file(
                file.joinpath("processed/data.csv"),
                window_size=self.window_size,
                stride=self.stride,
            )
            self.acc.extend(x)
            self.gyr.extend(x)
            self.l_acc.extend(y)
            self.l_gyr.extend(y)
            self.L.extend(length)

        self.acc = torch.cat(self.acc, dim=0)[:, :3].cumsum(dim=-1) * 0.01
        self.gyr = torch.cat(self.gyr, dim=0)[:, 3:6]
        self.l_acc = torch.cat(self.l_acc, dim=0)[:, :3].cumsum(dim=-1) * 0.01
        self.l_gyr = torch.cat(self.l_gyr, dim=0)[:, 3:6]

        self.L = torch.cat(self.L, dim=0)
        mask = torch.arange(self.acc.shape[-1]).expand(
            self.acc.shape[0], self.acc.shape[-1]
        )
        mask = mask < self.L.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(-1, self.acc.shape[1], -1)
        print(self.acc.shape, self.gyr.shape, self.l_acc.shape)
        self.acc = self.acc[mask]
        self.gyr = self.gyr[mask]
        self.l_acc = self.l_acc[mask]
        self.l_gyr = self.l_gyr[mask]
        print(self.acc.shape, self.gyr.shape, self.l_acc.shape)

        self.acc_mean = self.acc.mean().cpu().numpy()
        self.acc_std = self.acc.std().cpu().numpy()
        self.gyr_mean = self.gyr.mean().cpu().numpy()
        self.gyr_std = self.gyr.std().cpu().numpy()
        self.l_acc_mean = self.l_acc.mean().cpu().numpy()
        self.l_acc_std = self.l_acc.std().cpu().numpy()
        self.l_gyr_mean = self.l_gyr.mean().cpu().numpy()
        self.l_gyr_std = self.l_gyr.std().cpu().numpy()

        df = pd.DataFrame(
            {
                "acc_mean": [self.acc_mean],
                "acc_std": [self.acc_std],
                "gyr_mean": [self.gyr_mean],
                "gyr_std": [self.gyr_std],
                "l_acc_mean": [self.l_acc_mean],
                "l_acc_std": [self.l_acc_std],
                "l_gyr_mean": [self.l_gyr_mean],
                "l_gyr_std": [self.l_gyr_std],
            }
        )
        print(cl.Fore.green)
        print(df)
        print(cl.Style.reset)
        df.to_csv(self.root.joinpath("statics.csv"))

    def read_list(self, file):
        with open(file, "r") as f:
            files = f.readlines()
            files = [self.root.joinpath("datasets/data_publish_v2/" + f.strip()) for f in files]
        return files

    def load_files(self):
        self.dataX = []
        self.dataY = []
        self.dataL = []
        for file in tqdm(self.files):
            x, y, length = self._load_single_file(
                file.joinpath("processed/data.csv"),
                window_size=self.window_size,
                stride=self.stride,
            )
            self.dataX.extend(x)
            self.dataY.extend(y)
            self.dataL.extend(length)
        self.dataX = np.concatenate(
            [_.detach().cpu().numpy() for _ in self.dataX], axis=0
        )
        self.dataY = np.concatenate(
            [_.detach().cpu().numpy() for _ in self.dataY], axis=0
        )
        self.dataL = np.concatenate(
            [_.detach().cpu().numpy() for _ in self.dataL], axis=0
        )
        self.save()

    def _load_single_file(self, file, window_size: int, stride: int, return_wave=False):
        df = pd.read_csv(file)
        df.set_index("time", inplace=True)
        df.index = df.index / 1e9
        df.index = pd.to_datetime(df.index, unit="s")
        df = df.resample("5ms").mean()
        df = df.ffill()

        acc = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
        acc_norm = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
        acc_norm = np.linalg.norm(acc_norm, axis=-1)
        gyr = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        mag = df[["magnet_x", "magnet_y", "magnet_z"]].to_numpy()
        orientation = df[["ori_x", "ori_y", "ori_z", "ori_w"]].to_numpy()
        location = df[["pos_x", "pos_y", "pos_z"]].to_numpy()
        grav = df[["grav_x", "grav_y", "grav_z"]].to_numpy()


        mag = mag / 100
        acc, gyr, mag = rotateToWorldFrame(
            acc, gyr, mag, rotation=orientation, sample_rate=200.0
        )
        acc = GravityRemoval(acc=acc, location="RIDI", rescale=False)
        
        X, Y, Length = [], [], []
        for ii in range(2):
            acc_ = torch.from_numpy(acc).float()[ii::2]
            gyr_ = torch.from_numpy(gyr).float()[ii::2]
            mag_ = torch.from_numpy(mag).float()[ii::2]
            location_ = torch.from_numpy(location).float()[ii::2]
            orientation_ = orientation[ii::2]
            
            acc_norm_ = acc_norm[ii::2]
            orientation_ = self._orientation2degpersec(orientation_)

            velocity_, acceleration_ = self.vel_acc_generator(
                location=location_, sample_rate=100.0
            )
            valleys, acc_norm_ = self._step_finder(acc_norm=acc_norm_, window_size=5)
            acc_, gyr_, mag_, orientation_, velocity_, acceleration_ = (
                self._time_series_filter(
                    30, acc_, gyr_, mag_, orientation_, velocity_, acceleration_
                )
            )
            x, y, length = self.split_by_step(
                stepIdx=valleys,
                acc=acc_,
                gyr=gyr_,
                mag=mag_,
                orientation=orientation_,
                velocity=velocity_,
                acceleration=acceleration_,
                window_size=window_size,
                stride=stride,
                modes=self.mode,
            )

            X.append(x)
            Y.append(y)
            Length.append(length)
        return X, Y, Length


class ADVIODataset(mDataset):
    def __init__(
        self,
        root,
        mode="train",
        transform: Sequential = None,
        window_size=100,
        stride=10,
        keep_filters: list = None,
        skip_filters: list = None,
    ):
        super().__init__(
            root, mode, transform, window_size, stride, keep_filters, skip_filters
        )
        self.files = self.read_list(self.root.joinpath(f"{mode}_list.txt"))
        if self.check_existence():
            self.load(Path(self.root).joinpath(self.get_path_format()))
        else:
            self.load_files()

    def compute_statics(self):
        self.acc = []
        self.gyr = []
        self.l_acc = []
        self.l_gyr = []
        self.L = []

        lists = ["test", "train", "val"]
        files = []
        for mode in lists:
            files += self.read_list(self.root.joinpath(f"{mode}_list.txt"))

        for file in tqdm(files):
            x, y, length = self._load_single_file(
                file,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.acc.append(x[:, :3])
            self.gyr.append(x[:, 3:6])
            self.l_acc.append(y[:, :3])
            self.l_gyr.append(y[:, 3:6])
            self.L.append(length)

        self.acc = torch.cat(self.acc, dim=0).cumsum(dim=-1) * 0.01
        self.gyr = torch.cat(self.gyr, dim=0)
        self.l_acc = torch.cat(self.l_acc, dim=0).cumsum(dim=-1) * 0.01
        self.l_gyr = torch.cat(self.l_gyr, dim=0)
        self.L = torch.cat(self.L, dim=0)
        mask = torch.arange(self.acc.shape[-1]).expand(
            self.acc.shape[0], self.acc.shape[-1]
        )
        mask = mask < self.L.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(-1, self.acc.shape[1], -1)
        print(self.acc.shape, self.gyr.shape, self.l_acc.shape)
        self.acc = self.acc[mask]
        self.gyr = self.gyr[mask]
        self.l_acc = self.l_acc[mask]
        self.l_gyr = self.l_gyr[mask]
        print(self.acc.shape, self.gyr.shape, self.l_acc.shape)

        self.acc_mean = self.acc.mean().cpu().numpy()
        self.acc_std = self.acc.std().cpu().numpy()
        self.gyr_mean = self.gyr.mean().cpu().numpy()
        self.gyr_std = self.gyr.std().cpu().numpy()
        self.l_acc_mean = self.l_acc.mean().cpu().numpy()
        self.l_acc_std = self.l_acc.std().cpu().numpy()
        self.l_gyr_mean = self.l_gyr.mean().cpu().numpy()
        self.l_gyr_std = self.l_gyr.std().cpu().numpy()

        df = pd.DataFrame(
            {
                "acc_mean": [self.acc_mean],
                "acc_std": [self.acc_std],
                "gyr_mean": [self.gyr_mean],
                "gyr_std": [self.gyr_std],
                "l_acc_mean": [self.l_acc_mean],
                "l_acc_std": [self.l_acc_std],
                "l_gyr_mean": [self.l_gyr_mean],
                "l_gyr_std": [self.l_gyr_std],
            }
        )
        print(cl.Fore.green)
        print(df)
        print(cl.Style.reset)
        df.to_csv(self.root.joinpath("statics.csv"))

    def read_list(self, file):
        with open(file, "r") as f:
            files = f.readlines()
            files = [self.root.joinpath(f.strip()) for f in files]
        return files

    def load_files(self):
        self.dataX = []
        self.dataY = []
        self.dataL = []
        for file in tqdm(self.files):
            x, y, length = self._load_single_file(
                file,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.dataX.append(x.detach().cpu().numpy())
            self.dataY.append(y.detach().cpu().numpy())
            self.dataL.append(length.detach().cpu().numpy())
        self.dataX = np.concatenate(self.dataX, axis=0)
        self.dataY = np.concatenate(self.dataY, axis=0)
        self.dataL = np.concatenate(self.dataL, axis=0)

        self.save()

    def _load_single_file(self, file, window_size: int, stride: int, return_wave=False):
        df_acc = pd.read_csv(
            file.joinpath("iphone/accelerometer.csv"),
            header=None,
            names=["timestamp", "acc.X", "acc.Y", "acc.Z"],
        )
        df_gyr = pd.read_csv(
            file.joinpath("iphone/gyro.csv"),
            header=None,
            names=["timestamp", "gyr.X", "gyr.Y", "gyr.Z"],
        )
        df_mag = pd.read_csv(
            file.joinpath("iphone/magnetometer.csv"),
            header=None,
            names=["timestamp", "mag.X", "mag.Y", "mag.Z"],
        )
        df_loc = pd.read_csv(
            file.joinpath("ground-truth/pose.csv"),
            header=None,
            names=[
                "timestamp",
                "pose.X",
                "pose.Y",
                "pose.Z",
                "orientation.W",
                "orientation.X",
                "orientation.Y",
                "orientation.Z",
            ],
        )
        df_acc["timestamp"] = pd.to_timedelta(df_acc["timestamp"], unit="s")
        df_gyr["timestamp"] = pd.to_timedelta(df_gyr["timestamp"], unit="s")
        df_mag["timestamp"] = pd.to_timedelta(df_mag["timestamp"], unit="s")
        df_loc["timestamp"] = pd.to_timedelta(df_loc["timestamp"], unit="s")
        df_acc.set_index("timestamp", inplace=True)
        df_gyr.set_index("timestamp", inplace=True)
        df_mag.set_index("timestamp", inplace=True)
        df_loc.set_index("timestamp", inplace=True)
        df_merge = pd.merge(df_acc, df_gyr, how="outer", on="timestamp")
        df_merge = pd.merge(df_merge, df_mag, how="outer", on="timestamp")
        df_merge = pd.merge(df_merge, df_loc, how="outer", on="timestamp")
        df_merge.interpolate(method="index", inplace=True, limit_direction="forward")
        df_merge = df_merge.resample("10ms").mean()
        df_merge.ffill(inplace=True)

        acc = df_merge[["acc.X", "acc.Y", "acc.Z"]].to_numpy()
        gyr = df_merge[["gyr.X", "gyr.Y", "gyr.Z"]].to_numpy()
        mag = df_merge[["mag.X", "mag.Y", "mag.Z"]].to_numpy()
        location = df_merge[["pose.X", "pose.Y", "pose.Z"]].to_numpy()
        orientation = df_merge[
            ["orientation.X", "orientation.Y", "orientation.Z", "orientation.W"]
        ].to_numpy()
        mag = mag / 1000

        acc, gyr, mag = rotateToWorldFrame(
            acc,
            gyr,
            mag,
            rotation=orientation,
        )
        acc = np.roll(acc, shift=1, axis=1)
        gyr = np.roll(gyr, shift=1, axis=1)
        mag = np.roll(mag, shift=1, axis=1)
        location = np.roll(location, shift=1, axis=1)
        orientation = np.roll(R.from_quat(orientation).as_euler("xyz"), shift=1, axis=1)
        orientation = R.from_euler("xyz", orientation).as_quat()

        acc_norm = np.linalg.norm(acc, axis=-1)

        acc = GravityRemoval(acc, "advio")

        acc, gyr, mag, location, acc_norm, orientation = self._first_motion(
            acc_norm,
            0.5,
            acc,
            gyr,
            mag,
            location,
            acc_norm,
            orientation,
        )
        valleys, acc_norm = self._step_finder(acc_norm, window_size=5)

        acc = torch.from_numpy(acc).float().cuda()
        gyr = torch.from_numpy(gyr).float().cuda()
        mag = torch.from_numpy(mag).float().cuda()

        velocity, acceleration = self.vel_acc_generator(location, sample_rate=100.0)
        orientation = self._orientation2degpersec(orientation)

        assert (
            acc.shape
            == gyr.shape
            == mag.shape
            == location.shape
            == velocity.shape
            == acceleration.shape
            == orientation.shape
        )
        acc, gyr, mag, orientation, velocity, acceleration = self._time_series_filter(
            30, acc, gyr, mag, orientation, velocity, acceleration
        )
        x, y, length = self.split_by_step(
            stepIdx=valleys,
            acc=acc,
            gyr=gyr,
            mag=mag,
            orientation=orientation,
            velocity=velocity,
            acceleration=acceleration,
            window_size=window_size,
            stride=stride,
            modes=self.mode,
        )

        return x, y, length


class odomDataModule(pl.LightningDataModule):
    def __init__(self, config, data_dir: str = "./datasets") -> None:
        super().__init__()
        self.config = config
        self.data_dir = data_dir

    def gen_dataset(self, mode, dataset: str = None):
        if mode == "train":
            if self.config["pre_augmentation"]["all_direction"]:
                config = {
                    "probability": 1.0,
                    "mode": self.config["pre_augmentation"]["mode"],
                    "label_transform": True,
                    # "degree": 5,
                }
                transform = torch.nn.Sequential(
                    rotationNoise(config),
                )
            else:
                transform = None
        else:
            transform = None

        if dataset is not None:
            print(cl.Fore.red + f"Using {dataset} forcedly" + cl.Style.reset)
        else:
            dataset = self.config["dataset"]

        print(cl.Fore.green + f"Using {dataset} dataset" + cl.Style.reset)
        if dataset == "ADVIO":
            return ADVIODataset(
                root=self.data_dir,
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                transform=transform,
            )
        elif dataset == "OIOD":
            return OIODDataset(
                root=self.data_dir,
                mode=mode,
                window_size=self.config["window_size"],
                skip_filters=["nexus", "tango"],
                stride=self.config["stride"],
                transform=transform,
            )
        elif dataset == "OIOD_tango":
            return OIODDataset(
                root=self.data_dir.replace("OIOD_tango", "OIOD"),
                mode=mode,
                window_size=self.config["window_size"],
                keep_filters=["tango"],
                stride=self.config["stride"],
                transform=transform,
            )
        elif dataset == "RIDI":
            return RIDIDataset(
                root=self.data_dir,
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                transform=transform,
            )
        elif dataset == "RoNIN":
            return RoNINDataset(
                root=self.data_dir,
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                transform=transform,
            )
        elif dataset == "hybrid":
            OIOD_Tango = OIODDataset(
                root=self.data_dir.replace("hybrid", "OIOD"),
                mode=mode,
                window_size=self.config["window_size"],
                keep_filters=["tango"],
                stride=self.config["stride"],
                transform=transform,
            )
            RIDI = RIDIDataset(
                root=self.data_dir.replace("hybrid", "RIDI"),
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                transform=transform,
            )
            RoNIN = RoNINDataset(
                root=self.data_dir.replace("hybrid", "RoNIN"),
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                transform=transform,
            )
            ADVIO = ADVIODataset(
                root=self.data_dir.replace("hybrid", "ADVIO"),
                mode=mode,
                window_size=self.config["window_size"],
                stride=self.config["stride"],
                transform=transform,
            )
            return torch.utils.data.ConcatDataset([OIOD_Tango, RIDI, RoNIN, ADVIO])
        else:
            raise ValueError(f"Invalid dataset {self.config['dataset']}")

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            if hasattr(self, "train_dataset") and hasattr(self, "val_dataset"):
                return
            self.train_dataset = self.gen_dataset("train")
            self.val_dataset = self.gen_dataset("val")
        elif stage == "test":
            if hasattr(self, "test_dataset"):
                return
            self.test_dataset = self.gen_dataset("test")
        elif stage == "pred":
            if hasattr(self, "pred_dataset"):
                return
            self.pred_dataset = self.gen_dataset("pred")
        else:
            raise ValueError(f"Invalid stage {stage}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=os.cpu_count() // 2,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=os.cpu_count() // 2,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=os.cpu_count() // 2,
        )

    def pred_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pred_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=os.cpu_count() // 2,
        )


class MotionIDDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None) -> None:
        super().__init__()
        self.config = config
        self.root = Path("/mnt/Volume01/MotionID_Processed")
        self.window_size = self.config["window_size"]
        self.files = list(Path(self.root).glob("*.csv"))
        self.files = [Path(file) for file in self.files if file.stat().st_size < 1e9]
        self.files.sort()

        if self.check_existence():
            self.load()
        else:
            self.data = []
            for file in tqdm(self.files):
                self.data.extend(self.load_single_file(file))
            print(
                cl.Fore.green,
                f"Loaded {len(self.data)} files for MotionID.",
                cl.Style.reset,
            )
            self.save()

    def check_existence(self):
        return (self.root / f"total_{self.window_size}.pt").exists()

    def _get_storage_name(self):
        return self.root / f"total_{self.window_size}.pt"

    def save(self):
        torch.save(self.data, self._get_storage_name())

    def load(self):
        self.data = torch.load(
            self._get_storage_name(),
            mmap=True,
            weights_only=False,
            map_location="cpu",
        )

    def load_single_file(self, file):
        df = pd.read_csv(file)
        max_keep = len(df) // self.window_size * self.window_size
        date = df["timestamp"][0]
        acc = df[["acc.X", "acc.Y", "acc.Z"]].to_numpy()[0:max_keep]
        gyr = df[["gyr.X", "gyr.Y", "gyr.Z"]].to_numpy()[0:max_keep]
        mag = df[["mag.X", "mag.Y", "mag.Z"]].to_numpy()[0:max_keep]

        acc = GravityRemoval(acc=acc, location="MotionID", rescale=False, date=date)

        observation = np.concatenate([acc, gyr, mag], axis=1)
        observation = np.split(
            observation, len(observation) // self.window_size, axis=0
        )
        observation = np.stack(observation, axis=0).transpose(0, 2, 1)
        observation = [
            torch.tensor(
                o,
                dtype=torch.float32,
            )
            for o in observation
        ]

        return observation

    def __getitem__(self, idx):
        return (
            torch.rand_like(self.data[idx][:6]),
            self.data[idx][:6],
            torch.ones(1, dtype=torch.float32) * self.data[idx].shape[-1],
        )

    def __len__(self):
        return len(self.data)


class MotionIDModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.pin_memory = True
        self.num_workers = int(
            torch.get_num_threads()
            / torch.cuda.device_count()
        )
        self.shuffle = True
        config = {
            "probability": 1.0,
            "mode": "XY",
            "label_transform": True,
        }
        tranform = torch.nn.Sequential(
            rotationNoise(config),
        )
        self.datasets = MotionIDDataset(config=self.config, transform=tranform)
        self.setup("fit")

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            print("Loading train and val dataset.")
            self.train, self.val = torch.utils.data.random_split(
                self.datasets, [0.8, 0.2]
            )
            print(f"Loaded {len(self.train)} train samples.")
            print(f"Loaded {len(self.val)} val samples.")
            print("Completed splitting train and val dataset.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.config["batch_size"],
            pin_memory=self.config["pin_memory"],
            num_workers=int(self.num_workers),
            shuffle=self.config["shuffle"],
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.config["batch_size"],
            pin_memory=self.config["pin_memory"],
            num_workers=int(self.num_workers),
            shuffle=False,
        )


class OIODActionDatasets(OIODDataset):
    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        window_size=100,
        stride=10,
        keep_filters=None,
        skip_filters=None,
    ):
        super().__init__(
            root, mode, transform, window_size, stride, keep_filters, skip_filters
        )
        self.pseudo_label = torch.from_numpy(self.dataY)
        self.pseudo_label = (
            self.pseudo_label[:, :2].sum(dim=1).norm(dim=1) * 10 // 1 * 10
            + self.classes
        ).long()
        self.pseudo_label = self.pseudo_label
        self.pseudo_label = self.pseudo_label.cpu().numpy()
        self.classes_type = list(set(self.pseudo_label.tolist()))
        self.classes_type.sort()
        self.num_classes = 9
        print(cl.Fore.red, f"Number Pseudo Label : {self.num_classes}", cl.Style.reset)
        self.pseudo_label = torch.tensor(
            [self.classes_type.index(c) for c in self.pseudo_label]
        ).long()

    def __getitem__(self, idx):
        if isinstance(self.dataX[idx], np.ndarray):
            x = torch.from_numpy(self.dataX[idx]).float()
            y = torch.from_numpy(self.dataY[idx]).float()
        else:
            x = self.dataX[idx].float().clonse().detach()
            y = self.dataY[idx].float().clonse().detach()

        if self.transform is not None:
            x, y = self.transform((x, y))

        return x[:6], y, self.dataL[idx], self.classes[idx]
