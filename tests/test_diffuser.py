import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np

from acoustician_tools.diffuser import *


class TestDiffuser(unittest.TestCase):
    def test_qrd_parameters(self):
        expected = {
            'design_frequency': 1000,
            'generator': '7+0',
            'inverse': False,
            'low_frequency_diffusion_limit': 1000,
            'low_frequency_scatter_limit': 500,
            'high_cutoff_frequency': {
                '0°': 3648.0,
                '15°': 3524.0,
                '30°': 3160.0,
                '45°': 2580.0,
                '60°': 1824.0,
                '75°': 945.0,
                '90°': 0.0,
            },
            'depth_sequence': [0.0, 24.5, 98.0, 49.0, 49.0, 98.0, 24.5],
            'max_depth': 98.0,
            'well_width': 47,
            'separator_width': 3,
            'period_width': 350,
            'critical_distance': 1.03,
        }
        calculated = qrd_diffuser_parameters(f_design=1000, sep_w=3, n=7, m=0, inverse=False, c=343)
        self.assertDictEqual(calculated, expected)
