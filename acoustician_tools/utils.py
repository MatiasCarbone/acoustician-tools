"""
UTILS

This module contains various useful functions for calculating acoustic parameters to be used
in other modules and functions.
"""

import numpy as np
import math


def sound_speed(temperature: float = 20.0) -> float:
    """
    Calculate approximate speed of sound in dry air at a given temperature.

    Parameters:
        temperature (float): air temperature [°C]

    Returns:
        c (float): speed of sound [m/s]
    """
    c = 331.3 * math.sqrt(1 + (temperature / 273.15))
    return round(c, ndigits=1)
