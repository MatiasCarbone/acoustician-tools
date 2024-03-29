import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np
from acoustician_tools.room import *


class TestRoom(unittest.TestCase):
    def setUp(self):
        self.dimensions = [4, 3, 2]
        self.alpha = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2])
        self.alpha_multiband = np.array(
            [
                [0.2, 0.2, 0.3, 0.3, 0.1, 0.1],
                [0.5, 0.5, 0.5, 0.5, 0.2, 0.2],
                [0.7, 0.7, 0.6, 0.6, 0.3, 0.3],
            ]
        )
        self.volume = np.prod(self.dimensions)
        self.surfaces = shoebox_surfaces(*self.dimensions)

    def test_sound_speed(self):
        expected = 343.21
        self.assertAlmostEqual(sound_speed(20.0), expected, places=1)

    def test_rt_constant(self):
        sound_speed = 343.0
        expected = 0.161
        self.assertAlmostEqual(rt_constant(sound_speed, decay_db=60), expected, places=3)
        pass

    def test_shoebox_surfaces(self):
        expected = [8, 8, 6, 6, 12, 12]
        self.assertEqual(shoebox_surfaces(*self.dimensions), expected)

    def test_sabine(self):
        # Single band
        expected = 0.21
        self.assertAlmostEqual(
            rt_sabine(self.volume, self.surfaces, self.alpha),
            expected,
            places=2,
        )

        # Multiple bands
        expected = np.asarray([0.42, 0.21, 0.15])
        np.testing.assert_almost_equal(
            rt_sabine(self.volume, self.surfaces, self.alpha_multiband),
            expected,
            decimal=2,
        )

    def test_eyring(self):
        # Single band
        expected = 0.17
        self.assertAlmostEqual(
            rt_eyring(self.volume, self.surfaces, self.alpha),
            expected,
            places=2,
        )

        # Multiple bands
        expected = np.asarray([0.38, 0.17, 0.11])
        np.testing.assert_almost_equal(
            rt_eyring(self.volume, self.surfaces, self.alpha_multiband),
            expected,
            decimal=2,
        )

    def test_millington(self):
        # Single band
        expected = 0.156
        self.assertAlmostEqual(
            rt_millington(self.volume, self.surfaces, self.alpha),
            expected,
            places=2,
        )

        # Multiple bands
        expected = np.asarray([0.38, 0.156, 0.11])
        np.testing.assert_almost_equal(
            rt_millington(self.volume, self.surfaces, self.alpha_multiband),
            expected,
            decimal=2,
        )


if __name__ == '__main__':
    unittest.main()
