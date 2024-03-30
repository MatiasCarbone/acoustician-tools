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


def shoebox_surfaces(length: float, width: float, height: float) -> list:
    """
    Get list of each boundary's surface for a shoebox room with defined dimensions.

    Parameters:
        length (float): length of the room [m]
        width (float): width of the room [m]
        heigth (float): heigth of the room [m]

    Returns:
        surfaces (list of floats): list containig the surface of each wall [m2]
    """
    sidewalls = length * height
    front_rear = width * height
    roof_ceil = width * length

    surfaces = [sidewalls, sidewalls, front_rear, front_rear, roof_ceil, roof_ceil]
    return surfaces
