"""
RIR (ROOM IMPULSE RESPONSE)

This module contains functions for loading and processing room impulse-responses
and make various acoustical calculations.
"""

import numpy as np
from scipy.io import wavfile
from scipy.stats import linregress
from acoustician_tools.filter import butter_bandpass, butter_bandpass_filter


def clarity_from_ir(path: str, bands: list, t_early: int = 50):
    """
    Calculate clarity parameter from an impulse-response in .wav format.

    Parameters:
        path (string): Path to file. Must be a .wav audio file containing
            an impulse-response. Can be any bit sample-rate and bit depth
        bands (list): List of tuples, containing frequency bands (lower, upper) [Hz]
        t_early (int): Early time limit for early/late energy; [ms]
            (50ms for C50 and 80ms for C80 standards)

    Returns:
        clarity (list): List containing clarity values for each band
    """
    sr, y = wavfile.read(path)
    start = np.where(y > 0)[0][0]  # First non-zero value
    y = y[start:]  # Remove leading zeroes
    t = int((t_early / 1000) * sr)

    clarity = []
    for b in bands:
        y_filter = butter_bandpass_filter(y, b[0], b[1], sr, order=5)  # Bandpassed signal
        y_sq = y_filter**2
        c = 10 * np.log10(np.sum(y_sq[:t]) / np.sum(y_sq[t:]))
        clarity.append(c)
    return np.round(clarity, decimals=6).tolist()


def definition_from_ir(path: str, bands: list, t_early: int = 50):
    """
    Calculate definition parameter from an impulse-response in .wav format.

    Parameters:
        path (string): Path to file. Must be a .wav audio file containing
            an impulse-response. Can be any bit sample-rate and bit depth
        bands (list): List of tuples, containing frequency bands (lower, upper) [Hz]
        t_early (int): Early time limit for early/total energy; [ms]
            (50ms for D50 and 80ms for D80 standards)

    Returns:
        definition (list): List containing definition values for each band
    """
    sr, y = wavfile.read(path)
    start = np.where(y > 0)[0][0]  # First non-zero value
    y = y[start:]  # Remove leading zeroes
    t = int((t_early / 1000) * sr)

    definition = []
    for b in bands:
        y_filter = butter_bandpass_filter(y, b[0], b[1], sr, order=5)  # Bandpassed signal
        y_sq = y_filter**2
        d = 10 * np.log10(np.sum(y_sq[:t]) / np.sum(y_sq))
        definition.append(d)
    return np.round(definition, decimals=6).tolist()


def rt60_from_ir(path: str, bands: list, estimator: str = 't30'):
    """
    Get RT60 from a .wav impulse-response file.

    Parameters:
        path (string): Path to file. Must be a .wav audio file containing
            an impulse-response. Can be any bit sample-rate and bit depth
        bands (list): List of tuples, containing frequency bands (lower, upper) [Hz]
            Usually, RT60 is calculated using octave or third-octave bands
        estimator (string): Measurement range to be used to determine the RT60 using
            only a limited dynamic-range. [edt, t20, t30, t60]

    Returns:
        rt60 (list): List containing RT60 values for each frequency band [s]
    """
    sr, ir_signal = wavfile.read(path)
    rt60 = []

    # Get reference decay points based on selected estimator
    estimator = str.lower(str(estimator))
    match estimator:
        case 'edt':
            drop = (0, -10)
            multiplier = 6
        case 't10':
            drop = (-5, -15)
            multiplier = 6
        case 't20':
            drop = (-5, -25)
            multiplier = 3
        case 't30':
            drop = (-5, -35)
            multiplier = 2
        case 't60':
            drop = (-5, -65)
            multiplier = 1
        case _:
            raise TypeError('Invalid estimator. Only valid options are "edt", "t10", "t20", "t30" and "t60".')

    for b in bands:
        y = butter_bandpass_filter(ir_signal, b[0], b[1], sr, order=8)  # Bandpassed signal
        y_abs = np.abs(y) / np.max(np.abs(y))  # Absolute values, normalized

        # Scroeder integration
        sch = np.cumsum(y_abs[::-1] ** 2)[::-1]  # Backwards integration
        sch_db = sch_db = 10.0 * np.log10(sch / np.max(sch))  # Converted to dB

        # Reference decay x values points for slicing
        a = np.where(sch_db <= drop[0])[0][0]
        b = np.where(sch_db <= drop[1])[0][0]

        # Linear regression for segment of interest
        sch_db_slice = sch_db[a:b]
        t = np.linspace(0, (len(sch_db_slice) / sr), len(sch_db_slice))
        slope, intercept = linregress(t, sch_db_slice)[:2]

        # Calculate time multiplying linear regression
        regress_start = (drop[0] - intercept) / slope
        regress_end = (drop[1] - intercept) / slope
        rt60.append(multiplier * (regress_end - regress_start))

    return rt60
