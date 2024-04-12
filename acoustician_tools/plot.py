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
    time_range: int = 2000,
    slices: int = 10,
    window_length: int = 1024,
    window_shape: tuple = (0.25, 1.0),
    f_low: int = 20,
    f_high: int = 20000,
):
    # Load audio file
    sr, y = wavfile.read(path)
    peak = np.where(y == np.max(y))[0][0]

    # Generate Tukey window
    win_left = signal.windows.tukey(M=window_length, alpha=window_shape[0], sym=False)[: int(window_length / 2)]
    win_right = signal.windows.tukey(M=window_length, alpha=window_shape[1], sym=False)[int(window_length / 2) :]
    window = np.append(win_left, win_right)

    # Define stride and starting point for moving the window
    time_range_samples = int(300 * 0.001 * sr)
    stride = int(time_range_samples / (slices - 1))  # capaz -1
    start = int(peak - ((window_length / 2) * window_shape[0]))

    y = y[:time_range_samples]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(slices):
        # Pad the window to match the IR's length
        win = np.pad(window, (start, 0))
        try:
            win = np.pad(win, (0, len(y) - len(win)))
        except:
            break

        y_win = y * win

        # Calculate center frequency of bins for x-axis plotting
        n = len(y_win)
        if i == 0:
            xf = np.abs(np.fft.fftfreq(n, 1 / sr))
            left = np.where(xf >= f_low)[0][0]
            right = np.where(xf > f_high)[0][0]
            xf = xf[left:right]

        # Compute FFT of windowed audio
        yf = np.abs(np.fft.fft(y_win))
        yf = 20 * np.log10(yf)
        yf = yf[left:right]

        ax.plot(xf, yf, zs=time_range_samples - i, zdir='y', color='blue', alpha=0.5)
        start += stride

    plt.show()
