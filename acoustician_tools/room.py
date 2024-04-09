"""
ROOM

This module contains functions for calculating theoretical reverberation times of rooms, 
as well as calculating decay times and clarity from impulse-response files.
"""

import numpy as np

from acoustician_tools.utils import *

SOUNDSPEED = sound_speed(20.0)


def schroeder_frequency(t30: float, v: float):
    """
    Schroeder frequency (aka. transition frequency) calculation.

    Parameters:
        t30 (float): RT60 decay rime (measured as T30) [s]
        v (float): Volume of the room [m3]

    Returns:
        fs (float): Schroeder frequency [Hz]
    """
    fs = 2000 * np.sqrt(t30 / v)
    return round(fs, ndigits=2)


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


def rt_eyring(volume: float, surfaces: list, alphas: list, decay: int = 60, c: float = 343.0) -> float | list:
    """
    Calculate theoretical reverberation time using Eyring-Norris equation for one or more frequency bands.

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

    rt = constant * volume / (-total_surface * np.log(1 - mean_alpha))
    return rt


def rt_millington(volume: float, surfaces: list, alphas: list, decay: int = 60, c: float = 343.0) -> float | list:
    """
    Calculate theoretical reverberation time using Millington-Sette equation for one or more frequency bands.

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
    sigma = -np.sum(surfaces * np.log(1 - alphas), axis=-1)
    constant = rt_constant(c, decay)
    rt = constant * volume / sigma
    return rt


def rt_fitzroy(volume: float, surfaces: list, alphas: list, decay: int = 60, c: float = 343.0) -> float | list:
    """
    Calculate theoretical reverberation time using Fitzroy equation for one or more frequency bands.

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

    constant = rt_constant(c, decay)

    x, y, z = np.sum(surfaces[0:2]), np.sum(surfaces[2:4]), np.sum(surfaces[4:6])
    alpha_x = alphas.transpose()[0:2].sum(axis=0) / 2
    alpha_y = alphas.transpose()[2:4].sum(axis=0) / 2
    alpha_z = alphas.transpose()[4:6].sum(axis=0) / 2

    rt = (
        constant
        * (volume / np.sum(surfaces) ** 2)
        * ((-x / np.log(1 - alpha_x)) + (-y / np.log(1 - alpha_y)) + (-z / np.log(1 - alpha_z)))
    )
    return rt


def rt_arau(volume: float, surfaces: list, alphas: list, decay: int = 60, c: float = 343.0) -> float | list:
    """
    Calculate theoretical reverberation time using Arau-Puchades equation for one or more frequency bands.

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
    constant = rt_constant(c, decay)

    s_tot = np.sum(surfaces)
    sx, sy, sz = np.sum(surfaces[0:2]), np.sum(surfaces[2:4]), np.sum(surfaces[4:6])

    ax = np.average(alphas.transpose()[0:2], axis=0, weights=surfaces[0:2])
    ay = np.average(alphas.transpose()[2:4], axis=0, weights=surfaces[2:4])
    az = np.average(alphas.transpose()[4:6], axis=0, weights=surfaces[4:6])

    x_term = ((constant * volume) / -(s_tot * np.log(1 - ax))) ** (sx / s_tot)
    y_term = ((constant * volume) / -(s_tot * np.log(1 - ay))) ** (sy / s_tot)
    z_term = ((constant * volume) / -(s_tot * np.log(1 - az))) ** (sz / s_tot)

    rt = x_term * y_term * z_term
    return rt
