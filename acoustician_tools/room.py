"""
ROOM

This module contains functions for calculating theoretical reverberation times of rooms, 
as well as calculating decay times and clarity from impulse-response files.
"""

import numpy as np

SOUNDSPEED = 343.0  # m/s @ 20°C dry air


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


def t60_sabine(volume: float, surfaces: list, alphas: list, c: float = SOUNDSPEED) -> float | list:
    """
    Calculate theoretical reverberation time using Sabine's equation for one or more frequency bands.

    Parameters:
        volume (float): Total volume of the room [m3]
        surfaces (list of floats): Surface area of each boundary [m2]
        alphas (1d or 2d list of floats): Absortion coefficient for each boundary; [0-1]
            first dimension corresponds to boundary and second dimension to alpha

    Returns:
        t60 (float or list): Reverberation time in seconds. Amount of time required for a decay of 60dB
            for one at one or more frequency bands.
    """

    total_surface = np.sum(surfaces)
    mean_alpha = np.average(alphas, axis=-1, weights=surfaces)
    t60 = 0.161 * volume / (total_surface * mean_alpha)
    return t60


def t60_eyring(volume: float, surfaces: list, alphas: list, c: float = SOUNDSPEED) -> float | list:
    """
    Calculate theoretical reverberation time using Eyring-Norris equation for one or more frequency bands.

    Parameters:
        volume (float): Total volume of the room [m3]
        surfaces (list of floats): Surface area of each boundary [m2]
        alphas (1d or 2d list of floats): Absortion coefficient for each boundary; [0-1]
            first dimension corresponds to boundary and second dimension to alpha

    Returns:
        t60 (float or list): Reverberation time in seconds. Amount of time required for a decay of 60dB
            for one at one or more frequency bands.
    """

    total_surface = np.sum(surfaces)
    mean_alpha = np.average(alphas, axis=-1, weights=surfaces)
    t60 = 0.161 * volume / (-total_surface * np.log(1 - mean_alpha))
    return t60
