"""
ABSORBER

This module contains function for calculating acoustic parameters of absorbers and materials.
"""

import numpy as np
import math
from acoustician_tools.utils import sound_speed, air_density

SOUNDSPEED = sound_speed(20.0)
AIR_DENSITY = air_density(20.0, 1013)


def nrc(alphas: list) -> float:
    """
    Calculate Noise Reduction Coefficient from multi-band absorption.

    Parameters:
        alphas (list): List of absorption coefficients at 250, 500, 1000 and 2000 Hz [0-1]

    Returns:
        nrc (float): Noise reduction coefficient, rounded to nearest 0.05 [0-1]
    """
    if len(alphas) != 4:
        raise Exception('Input should be a list of 4 alpha coefficients [0-1].')
    elif np.min(alphas) < 0:
        raise Exception('Negative alpha coefficients are not possible.')

    nrc = np.round(np.average(alphas) * 20) / 20

    if nrc > 1.0:
        return 1.0
    else:
        return nrc
