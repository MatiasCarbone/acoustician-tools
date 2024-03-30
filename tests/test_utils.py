import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np
from acoustician_tools.utils import *


class TestUtils(unittest.TestCase):
    def test_sound_speed(self):
        expected = 343.21
        self.assertAlmostEqual(
            sound_speed(20.0),
            expected,
            places=1,
            msg='Sound speed at 20.0 °C is approximately 343m/s',
        )

    def test_shoebox_surfaces(self):
        expected = [8, 8, 6, 6, 12, 12]
        self.assertEqual(
            shoebox_surfaces(4, 3, 2),
            expected,
            msg='The boundary-surfaces for a 4x3x2 room should be 8-8-6-6-12-12',
        )

    def test_wavelength(self):
        arguments = [200, 343.0, 'm']
        expected = 1.715

        self.assertEqual(
            frequency_to_wavelength(*arguments),
            expected,
            msg='Wavelength for a sound wave of 200Hz @ 343m/s is 1.715m',
        )

        arguments = [10000, 355.0, 'mm']
        expected = 35.5

        self.assertEqual(
            frequency_to_wavelength(*arguments),
            expected,
            msg='Wavelength for a sound wave of 10000Hz @ 355m/s is 35.5mm',
        )

    def test_frequency(self):
        arguments = [5, 343.0, 'm']
        expected = 68.6

        self.assertEqual(
            wavelength_to_frequency(*arguments),
            expected,
            msg='Frequency for a wavelength of 5m @ 343m/s is 68.6Hz',
        )

        arguments = [4, 355.0, 'cm']
        expected = 8875

        self.assertEqual(
            wavelength_to_frequency(*arguments),
            expected,
            msg='Frequency for a wavelength of 4cm @ 355m/s is 8875Hz',
        )

    def test_air_density(self):
        arguments = [20.0, 1.013]
        expected = 1.204
        self.assertAlmostEqual(air_density(*arguments), expected, places=3)

        arguments = [0.0, 1.000]
        expected = 1.275
        self.assertAlmostEqual(air_density(*arguments), expected, places=3)


if __name__ == '__main__':
    unittest.main()
