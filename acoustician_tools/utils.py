"""
UTILS

This module contains various useful functions for calculating acoustic parameters to be used
in other modules and functions.
"""

import numpy as np
import math


def frequency_to_wavelength(frequency: float, c: float = 343.0, units: str = 'm') -> float:
    """
    Calculate wavelength of a given frequency.

    Parameters:
        frequency (float) [Hz]
        c (float): speed of sound [m/s]
        units (str): units to convert wavelength to [km, m, cm, mm]

    Returns:
        l (float): wavelength in selected unit [default: m]
    """

    conversion = {'km': 0.001, 'm': 1, 'cm': 100, 'mm': 1000}
    l = c / frequency * conversion[units]
    return l


def wavelength_to_frequency(wavelength: float, c: float = 343.0, units: str = 'm') -> float:
    """
    Calculate frequency of a given wavelength.

    Parameters:
        l (float): wavelength in selected unit [default: m]
        c (float): speed of sound [m/s]
        units (str): units to convert wavelength to [km, m, cm, mm]

    Returns:
        frequency (float) [Hz]
    """

    conversion = {'km': 0.001, 'm': 1, 'cm': 100, 'mm': 1000}
    f = c / wavelength * conversion[units]
    return f


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


def air_density(temperature: float = 20.0, pressure: float = 1.013) -> float:
    """
    Calculate density of dry air (0% relative humidity).

    Parameters:
        temperature (float): [°C]
        pressure (float): [bar]

    Returns:
        density (float): [kg/m3]
    """
    gas_constant = 287.058
    density = (pressure * 100000) / (gas_constant * (temperature + 273.15))
    return density


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


def cot(x):
    return 1 / np.tan(x)


def coth(x):
    return np.cosh(x) / np.sinh(x)
