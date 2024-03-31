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


def porous_absorber_alpha(
    thickness: float,
    flow_resistivity: int,
    frequencies: list = range(100, 20001, 50),
    c: float = 343,
    air_density: float = 1.204,
):
    """
    doi.org/10.1016/j.apacoust.2013.06.004
    """

    # Cotangent
    def cot(x):
        return 1 / np.tan(x)

    # Hyperbolic-cotangent
    def coth(x):
        return np.cosh(x) / np.sinh(x)

    # Convert frequencies to array
    f = np.array(frequencies)

    # Characteristic impedance of air [Pa.s/m2]
    z0 = c * air_density

    # Parameter for calculating Z and Gamma of absorbent layer [dimensionless]
    x = air_density * f / flow_resistivity

    # --------------------- Allard-Champoux calculations for z1 -------------------- #
    # Dynamic density for absorption layer [kg.m3]
    p = 1.2 + (-0.0364 * (x**-2) - 1j * 0.1144 * (x**-1)) ** 0.5

    # Dynamic bulk modulus
    k = (
        101320
        * (1j * 29.64 + (2.82 * (x**-2) + 1j + 24.9 * (x**-1)) ** 0.5)
        / (1j * 21.17 + (2.82 + (x**2) + 1j * 24.9 * (x**-1)) ** 0.5)
    )

    # Characteristic impedance of absorption layer
    z1 = (p * k) ** 0.5

    # Propagation constant for absorption layer
    gamma1 = 1j * 2 * np.pi * f * (p / k) ** 0.5

    # ----------------------- Transfer-Matrix Calculations ------------------------ #
    d1 = thickness * 0.001  # Thickness converted to meters
    zs2 = 1j * np.inf  # Surface impedance of rigid backwall !!!!!!!!!!!

    # Surface impedance for absorber layer
    zs1 = (z1 * zs2 * coth(gamma1 * d1) + (z1**2)) / (zs2 + z1 * coth(gamma1 * d1))

    # -------------------------- Absorption coefficient -------------------------- #
    # Complex reflection factor of system [dimensionless]
    r = (zs1 - z0) / (zs1 + z0)

    # Absorption coefficient at normal incidence
    alpha = 1 - np.abs(r) ** 2
