import colored as cl
import numpy as np
import torch
from scipy.signal import find_peaks


def find_peaks_and_valleys(filtered_wave):
    peaks, _ = find_peaks(filtered_wave - filtered_wave.mean())
    valleys, _ = find_peaks(-filtered_wave + filtered_wave.mean())
    return peaks, valleys


def find_steps(filtered_wave, tolerance=30):
    peaks, valleys = find_peaks_and_valleys(filtered_wave)

    while np.min(np.diff(peaks)) < tolerance:
        temp = []
        i = 0
        while i < len(peaks) - 1:
            if peaks[i + 1] - peaks[i] < tolerance:
                temp.append(
                    peaks[i + 1]
                    if filtered_wave[peaks[i + 1]] > filtered_wave[peaks[i]]
                    else peaks[i]
                )
                i += 1
            else:
                temp.append(peaks[i])
            i += 1
        peaks = np.array(temp)

    while np.min(np.diff(valleys)) < tolerance:
        temp = []
        i = 0
        while i < len(valleys) - 1:
            if valleys[i + 1] - valleys[i] < tolerance:
                temp.append(
                    valleys[i + 1]
                    if filtered_wave[valleys[i + 1]] < filtered_wave[valleys[i]]
                    else valleys[i]
                )
                i += 1
            else:
                temp.append(valleys[i])
            i += 1
        valleys = np.array(temp)

    for i in range(1, len(peaks)):
        temp = valleys[(valleys > peaks[i - 1]) & (valleys < peaks[i])]
        while len(temp) > 1:
            valleys = valleys[valleys != temp.max()]
            temp = temp[temp != temp.max()]

    for i in range(1, len(valleys)):
        temp = peaks[(peaks > valleys[i - 1]) & (peaks < valleys[i])]
        while len(temp) > 1:
            peaks = peaks[peaks != temp.min()]
            temp = temp[temp != temp.min()]

    missing_valleys = []
    for i in range(1, len(peaks)):
        temp = valleys[(valleys > peaks[i - 1]) & (valleys < peaks[i])]
        if len(temp) == 0:
            missing_valleys.append(i)
    for i in missing_valleys:
        valleys = np.insert(
            valleys,
            min(i, len(valleys)),
            np.argmin(filtered_wave[peaks[i - 1] : peaks[i]]) + peaks[i - 1],
        )
    missing_peaks = []
    for i in range(1, len(valleys)):
        temp = peaks[(peaks > valleys[i - 1]) & (peaks < valleys[i])]
        if len(temp) == 0:
            missing_peaks.append(i)

    if len(peaks) != len(valleys):
        if len(peaks) > len(valleys):
            peaks = peaks[:-1]
        else:
            valleys = valleys[:-1]

    return peaks, valleys
