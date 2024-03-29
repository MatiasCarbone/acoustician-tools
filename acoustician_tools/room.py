"""
ROOM

This module contains functions for calculating theoretical reverberation times of rooms, 
as well as calculating decay times and clarity from impulse-response files.
"""

import numpy as np
import math

SOUNDSPEED = 343.0  # m/s @ 20°C dry air


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


def rt_constant(sound_speed: float = SOUNDSPEED, decay_db: int = 60) -> float:
    """
    Calculate RT constant to be used in reverberation formulas.

    Parameters:
        sound_speed (float): speed of sound [m/s]
        decay_db (int): drop in level for calculating reverberation time; [dB]
            ex: 60 for RT60, 30 for RT30...

    Returns:
        constant (float): RT constant
    """
    constant = np.log(10 ** (decay_db / 10)) * 4 / sound_speed
    return constant


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


def rt_sabine(volume: float, surfaces: list, alphas: list, decay: int = 60, c: float = 343.0) -> float | list:
    """
    Calculate theoretical reverberation time using Sabine's equation for one or more frequency bands.

    Parameters:
        volume (float): Total volume of the room [m3]
        surfaces (list of floats): Surface area of each boundary [m2]
        alphas (1d or 2d list of floats): Absortion coefficient for each boundary; [0-1]
            first dimension corresponds to boundary and second dimension to alpha
        decay (int): intensity drop for computing reverberation time [dB]
            ex: 60 for RT60, 30 for RT30...
        c (float): speed of sound [m/s]

    Returns:
        rt (float or list): Reverberation time. Amount of time required for a decay [s]
            of specified amount of dB at one or more frequency bands.
    """

    total_surface = np.sum(surfaces)
    mean_alpha = np.average(alphas, axis=-1, weights=surfaces)
    constant = rt_constant(c, decay)

    rt = constant * volume / (total_surface * mean_alpha)
    return rt


def t60_eyring(volume: float, surfaces: list, alphas: list) -> float | list:
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


def t60_millington(volume: float, surfaces: list, alphas: list) -> float | list:
    """
    Calculate theoretical reverberation time using Millington-Sette equation for one or more frequency bands.

    Parameters:
        volume (float): Total volume of the room [m3]
        surfaces (list of floats): Surface area of each boundary [m2]
        alphas (1d or 2d list of floats): Absortion coefficient for each boundary; [0-1]
            first dimension corresponds to boundary and second dimension to alpha

    Returns:
        t60 (float or list): Reverberation time in seconds. Amount of time required for a decay of 60dB
            for one at one or more frequency bands.
    """
    sigma = -np.sum(surfaces * np.log(1 - alphas), axis=-1)
    t60 = 0.161 * volume / sigma
    return t60
