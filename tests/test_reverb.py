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

    def test_shoebox_surfaces(self):
        expected = [8, 8, 6, 6, 12, 12]
        self.assertEqual(shoebox_surfaces(*self.dimensions), expected)

    def test_sabine(self):
        # Single band
        expected = 0.21
        self.assertAlmostEqual(
            t60_sabine(self.volume, self.surfaces, self.alpha),
            expected,
            places=2,
        )

        # Multiple bands
        expected = np.asarray([0.42, 0.21, 0.15])
        np.testing.assert_almost_equal(
            t60_sabine(self.volume, self.surfaces, self.alpha_multiband),
            expected,
            decimal=2,
        )

    def test_eyring(self):
        # Single band
        expected = 0.17
        self.assertAlmostEqual(
            t60_eyring(self.volume, self.surfaces, self.alpha),
            expected,
            places=2,
        )

        # Multiple bands
        expected = np.asarray([0.38, 0.17, 0.11])
        np.testing.assert_almost_equal(
            t60_eyring(self.volume, self.surfaces, self.alpha_multiband),
            expected,
            decimal=2,
        )


if __name__ == '__main__':
    unittest.main()
