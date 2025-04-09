import datetime

import ahrs
import colored as cl
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as R


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def rotateToWorldFrame(
    acc: np.array,
    gyr: np.array,
    mag: np.array,
    rotation: np.array = None,
    cutoff: float = 50.0,
    name: str = "",
    sample_rate: float = 100.0,
    return_ori: bool = False,
) -> (np.array, np.array, np.array):
    if rotation is not None:
        quad = rotation
        r = R.from_quat(quad)
        inv = r.inv()
    else:
        filter = ahrs.filters.Madgwick(
            acc=acc,
            gyr=gyr,
            mag=mag / 10,
            frequency=sample_rate,
            # gain=0.1,
        )
        quad = filter.Q
        quad = np.concatenate((quad[:, 1:], quad[:, :1]), axis=1)
        r = R.from_quat(quad)
        inv = r.inv()

    acc = r.apply(acc)
    gyr = r.apply(gyr)
    mag = r.apply(mag)
    if return_ori:
        return acc, gyr, mag, r.as_quat()
    return acc, gyr, mag


def coordinate_exchange(datas: list, _from: str, _to: str):
    result = []
    for data in datas:
        if _from == "ENU" and _to == "NED":
            data = [data[:, 1], data[:, 0], -data[:, 2]]
        elif _from == "NED" and _to == "ENU":
            if np.mean(data[2]) < 0:
                print(
                    cl.Fore.red
                    + "The data might not be in NED coordinate system"
                    + cl.Style.reset
                )
            data = [data[:, 1], data[:, 0], -data[:, 2]]
        else:
            raise ValueError(
                f"The coordinate system is not supported from: {_from} to {_to}"
            )
        result.append(np.array(data).swapaxes(0, 1))
    return result
