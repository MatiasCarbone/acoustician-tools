"""
BANDS

This module contains functions for calculating frequency bands.
"""

import numpy as np


def octave_bands():
    """
    Generate a list of center and boundary frequencies for octave bands
    multiple of 1000Hz.
    Note - these are exact frequencies (rounded to 3 decimal), not
    standard (nominal) frequencies.

    Returns:
        bands (dict): Dictionary containing two lists of frequencies;
            f_center: List of center frequencies for each band
            f_bound: List of tuples, containing boundary (lower and
                upper) frequencies for each band
    """
    f0 = [1000]
    while min(f0) > 20:
        f0.insert(0, f0[0] / 2)
    while max(f0) < 15000:
        f0.append(f0[-1] * 2)

    flow = np.asarray(f0) / np.sqrt(2)
    fhigh = np.asarray(f0) * np.sqrt(2)

    bands = {
        'f_center': f0,
        'f_bound': list(zip(flow.round(3).tolist(), fhigh.round(3).tolist())),
    }
    return bands
