"""
DIFFUSER

This module contains functions for calculating acoustic parameters of various types of diffusers.
"""

import numpy as np
from acoustician_tools.utils import sound_speed, frequency_to_wavelength, wavelength_to_frequency

SOUNDSPEED = sound_speed(20.0)


def qrd_diffuser_parameters(
    f_design: float,
    sep_w: float,
    n: int,
    m: int = 0,
    width: float = 47,
    c: float = 343,
):
    # Convert frequency to wavelength [m]
    lambda_design = frequency_to_wavelength(f_design, c)

    # Generate quadratic-residue sequence
    range = np.arange(0, n, 1)
    sequence = ((range**2) + m) % n

    # Calculate depth for each well in the sequence [mm]
    d = np.round(((sequence * lambda_design) / (2 * n)) * 1000, decimals=2)
    d_max = max(d)

    # Minimum recommended well width [mm]
    if d_max >= 400:
        w_min = d_max / 16
    else:
        w_min = 25

    # Maximum recommended well width [mm]
    w_max = np.ceil(lambda_design * 0.25 * 1000)

    # Assign value to width it it was given, or else calculate it
    # according to Schroeder recommendations [mm]
    if width:
        w = width
    else:
        w = np.round(lambda_design * 0.137 * 1000, decimals=2)

    # Warnings for exceeding width dimensions
    if w < w_min:
        print(f'WARNING!: The selected width value is below the minimum value of {w_min}mm. Beware of viscous losses.')
    elif w > w_max:
        print(f"WARNING!: The selected width value is above the maximum value of {w_max}mm.")

    # Calculate period width (wells + separators) [mm]
    period = n * (w + sep_w)
    if period < (lambda_design / 2):
        print(
            f"WARNING!: The period width is shorter than required. The effective design frequency will be higher than expected."
        )
        f_low = wavelength_to_frequency(period)
    else:
        f_low = f_design

    # High-frequency limit (normal angle)
    f_high = int(wavelength_to_frequency((w * 2) / 1000, c))

    # High-frequency limit at various angles
    angles = np.arange(0, 91, 15)
    angles_radians = angles * (np.pi / 180)
    f_high_angles = np.ceil(f_high * np.sin(abs((90 * (np.pi / 180)) - angles_radians)))

    # Create a dictionary for high-frequency cutoff values
    keys = [str(x) + '°' for x in angles]
    f_high_dict = dict(zip(keys, f_high_angles))

    # Minimum seating distance [m]
    d_listen = np.round(3 * frequency_to_wavelength(f_low, c), decimals=2)

    params = {
        'design_frequency': f_design,
        'generator': f'{n}+{m}',
        'low_frequency_diffusion_limit': f_low,
        'low_frequency_scatter_limit': int(f_low / 2),
        'high_cutoff_frequency': f_high_dict,
        'depth_sequence': d.tolist(),
        'max_depth': d_max,
        'well_width': w,
        'separator_width': sep_w,
        'period_width': period,
        'critical_distance': d_listen,
    }

    # Plate frequency
    f_plate = f_design * n
    if f_high > f_plate:
        print(f'Plate frequency ({f_plate}Hz) is lower than cutoff frequency and is the new upper limit.')
        params['high_cutoff_frequency'] = f_plate

    return params
