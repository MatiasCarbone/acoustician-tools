"""
ABSORBER

This module contains functions for calculating acoustic parameters of absorbers and materials.
"""

import numpy as np
from acoustician_tools.utils import sound_speed, air_density, coth, cot

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


def porous_absorber(
    flow_resistivity: float,
    thickness: float,
    frequencies: list = range(100, 20001, 50),
    c: float = 343,
    air_density: float = 1.204,
):
    """
    Calculate the absortion coefficients of a layer of porous absorber agains a rigid baking with no
    air gap and normal incidence, at one or more individual frequencies.

    The calculations are performed using Delany and Bazley equations for impedance and wave number,
    following the instructions and formulas presented in
        'Trevor J. Cox and Peter D'Antonio. 2009.
        Acoustic Absorbers and Diffusers: Theory, design and application,
        2nd Edition. Taylor & Francis.'

    Parameters:
        flow_resistivity (float) [Pa.s/m2]
        thickness (float): total material thickness [mm]
        frequencies (float or list): one or more individual frequencies (not bands) [Hz]
            for absortion coefficient to be calculated at
        c (float): speed of sound [m/s]
        air_density (float) [kg/m3]

    Returns:
        alpha (np.array): array containing alpha coefficients for each frequency [0-1]
    """
    # Convert frequencies to array
    f_list = np.asarray(frequencies)

    z0 = c * air_density  # Characteristic impedance of air

    # Delany and Bazley calculation for impedance and wave number
    x = air_density * f_list / flow_resistivity  # Dimensionless quantity
    zc = air_density * c * (1 + 0.0571 * (x**-0.754) - 1j * 0.087 * (x**-0.732))  # Characteristic impedance of material
    k = (2 * np.pi / c) * f_list * (1 + 0.0978 * (x**-0.700) - 1j * 0.189 * (x**-0.595))  # Complex wave number

    # Absorption coefficients calculation
    l = thickness * 0.001  # Material thickness converted to meters
    z = -1j * zc * (1 / np.tan(k * l))  # Surface impedance
    r = (z - z0) / (z + z0)  # Reflection factor
    alpha = 1 - np.abs(r) ** 2  # Absortion coefficient at normal incidence

    # TODO: implement parameters for returning other acoustic parameters.
    return alpha


def helmholtz_resonant_frequency(
    opening_diameter: float,
    opening_length: float,
    cavity_dimensions: list,
    end_correction: float = 0.3,
    opening_shape: str = 'circle',
    cavity_shape: str = 'cylinder',
    c: float = 343,
):
    """
    Calculate the resonant frequency of a Helmholtz resonator with circle or square opening shape and
    cylindrical or prismatic cavity shape.

    Parameters:
        opening_diameter (float): Dimensions of the hole, diameter in case of circular hole [mm]
            or side in case of square hole
        opening_length (float): Length of the hole (neck) [mm]
        cavity_dimensions (list): Dimensions for the inner cavity. List of two float values; [mm, mm]
            first corresponds to diameter (side in case of a prismatic cavity) and second
            corresponds to depth of the cavity
        end_correction (float): Correction accounting for waves forming before the start of a
            tube; usually 0.3 for pipes with one opening and 0.6 for pipes with two
        c (float): Speed of sound [m/s]

    Return:
        f (float): Resonant frequency of the system [Hz]
    """

    # Calculate area of the opening for circle or square shape
    match opening_shape:
        case 'circle':
            a = np.pi * ((opening_diameter / 2) ** 2)
        case 'square':
            a = opening_diameter**2
        case _:
            raise Exception('The only valid opening shapes for this calculation as square and circle.')
    # Calculate volume of the cavity for cylinder or prism shape
    match cavity_shape:
        case 'cylinder':
            v = np.pi * ((cavity_dimensions[0] / 2) ** 2) * cavity_dimensions[1]
        case 'prism':
            v = (cavity_dimensions[0] ** 2) * cavity_dimensions[1]
        case _:
            raise Exception('The only valid cavity shapes for this calculation as cylinder and prism.')

    l = opening_length + end_correction * opening_diameter  # Length + optional end-correction [m]

    # Convert units to meters
    a = a * (10**-6)
    l = l * (10**-6)
    v = v * (10**-6)

    f = (c / (2 * np.pi)) * np.sqrt(a / (v * l))  # Resonant frequency [Hz]
    return np.round(f, decimals=3)
