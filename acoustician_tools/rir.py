"""
RIR (ROOM IMPULSE RESPONSE)

This module contains functions for loading and processing room impulse-responses
and make various acoustical calculations.
"""

import csv
import numpy as np


def load_ir_mono(path: str, sep: str = ','):
    """
    Load a mono IR from CSV or text file.
    """
    with open(path) as file:
        csv_reader = csv.reader(file, delimiter=sep)
        data = [float(x[0]) for x in csv_reader]
    return data
