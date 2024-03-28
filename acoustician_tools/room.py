"""
ROOM

This module contains functions for calculating theoretical reverberation times of rooms, 
as well as calculating decay times and clarity from impulse-response files.
"""

import numpy as np

SOUNDSPEED = 343.0  # m/s @ 20°C dry air


def t60_sabine(volume: float, surfaces: list[float], alphas: list[float], c: float = SOUNDSPEED) -> float:
    """
    Calculate theoretical reverberation time using Sabine's equation.

    Parameters:
        volume (float): Total volume of the room [m3]
        surfaces (list of floats): Surface area of each boundary [m2]
        alphas (list of floats): Absortion coefficient for each boundary [0-1]

    Returns:
        t60: Reverberation time in seconds. Amount of time required for a decay of 60dB
    """

    total_surface = np.sum(surfaces)
    mean_alpha = np.average(alphas, weights=surfaces)

    t60 = 0.161 * volume / (total_surface * mean_alpha)
    return t60
