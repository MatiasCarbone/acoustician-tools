import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np
from acoustician_tools.room import t60_sabine, shoebox_surfaces


class TestRoom(unittest.TestCase):
    def setUp(self):
        # Initialize properties here
        pass

    def test_shoebox_surfaces(self):
        expected = [8, 8, 6, 6, 12, 12]
        self.assertEqual(shoebox_surfaces(4, 3, 2), expected)

    def test_sabine(self):
        # Single band
        expected = 1.2113
        self.assertAlmostEqual(
            t60_sabine(3000, [240, 600, 500], [0.1, 0.25, 0.45]),
            expected,
            msg=f'Should be close to {expected}.',
            places=2,
        )

        # Multiple bands
        l, w, h = 2, 4, 3
        vol = l * w * h
        a = [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
        expected = np.asarray([0.14861538, 0.74307692])
        self.assertTrue(
            np.allclose(
                t60_sabine(vol, shoebox_surfaces(l, w, h), alphas=a),
                expected,
            )
        )


if __name__ == '__main__':
    unittest.main()
