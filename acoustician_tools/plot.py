"""
PLOT

This module contains functions for generating various plots from audio.
"""

import numpy as np
from scipy import signal
from scipy import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt


def waterfall_fft_plot(
    path: str,
    time_range: int = 500,
    slices: int = 31,
    window_length: int = 512,
    window_shape: tuple = (0.25, 1.0),
):
    sr, y = wavfile.read(path)
    peak = np.where(y == np.max(y))[0][0]
    stride = int(time_range / (slices - 1))

    win_left = signal.windows.tukey(M=window_length, alpha=window_shape[0], sym=False)[: int(window_length / 2)]
    win_right = signal.windows.tukey(M=window_length, alpha=window_shape[1], sym=False)[int(window_length / 2) :]
    window = np.append(win_left, win_right)

    start = int(peak - (window_length * window_shape[0]))
    end = start + window_length

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(slices):
        y_slice = y[start:end]
        y_window = y_slice * window

        n = len(y_window)
        yf = np.abs(np.fft.fft(y_window))
        yf = yf[: (n // 2) + 1]

        start += stride
        end += stride

        if i == 0:
            xf = np.linspace(0, sr / 2, (n // 2) + 1)

        ax.plot(xf, yf, zs=slices - i, zdir='y', color='blue', alpha=0.8)
