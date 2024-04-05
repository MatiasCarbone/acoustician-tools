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

        expected = expected = {
            'design_frequency': 612,
            'generator': '17+9',
            'inverse': False,
            'low_frequency_diffusion_limit': 612,
            'low_frequency_scatter_limit': 306,
            'high_cutoff_frequency': {
                '0°': 5044.0,
                '15°': 4873.0,
                '30°': 4369.0,
                '45°': 3567.0,
                '60°': 2522.0,
                '75°': 1306.0,
                '90°': 0.0,
            },
            'depth_sequence': [
                148.36,
                164.84,
                214.29,
                16.48,
                131.87,
                0.0,
                181.32,
                115.39,
                82.42,
                82.42,
                115.39,
                181.32,
                0.0,
                131.87,
                16.48,
                214.29,
                164.84,
            ],
            'max_depth': 214.29,
            'well_width': 34,
            'separator_width': 2,
            'period_width': 612,
            'critical_distance': 1.68,
        }
        calculated = qrd_diffuser_parameters(f_design=612, sep_w=2, width=34, n=17, m=9, inverse=False, c=343)
        self.assertDictEqual(calculated, expected)

        expected = {
            'design_frequency': 357,
            'generator': '17+3',
            'inverse': True,
            'low_frequency_diffusion_limit': 357,
            'low_frequency_scatter_limit': 178,
            'high_cutoff_frequency': {
                '0°': 3648.0,
                '15°': 3524.0,
                '30°': 3160.0,
                '45°': 2580.0,
                '60°': 1824.0,
                '75°': 945.0,
                '90°': 0.0,
            },
            'depth_sequence': [
                395.62,
                367.36,
                282.58,
                141.29,
                423.88,
                169.55,
                339.1,
                452.13,
                28.26,
                28.26,
                452.13,
                339.1,
                169.55,
                423.88,
                141.29,
                282.58,
                367.36,
            ],
            'max_depth': 452.13,
            'well_width': 47,
            'separator_width': 2,
            'period_width': 833,
            'critical_distance': 2.88,
        }
        calculated = qrd_diffuser_parameters(f_design=357, sep_w=2, n=17, m=3, inverse=True, c=343)
        self.assertDictEqual(calculated, expected)
