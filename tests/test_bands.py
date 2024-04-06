import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np

from acoustician_tools.bands import *


class TestBands(unittest.TestCase):
    def test_octave_bands(self):
        expected = {
            'f_center': [15.625, 31.25, 62.5, 125.0, 250.0, 500.0, 1000, 2000, 4000, 8000, 16000],
            'f_bound': [
                (11.049, 22.097),
                (22.097, 44.194),
                (44.194, 88.388),
                (88.388, 176.777),
                (176.777, 353.553),
                (353.553, 707.107),
                (707.107, 1414.214),
                (1414.214, 2828.427),
                (2828.427, 5656.854),
                (5656.854, 11313.708),
                (11313.708, 22627.417),
            ],
        }
        self.assertDictEqual(octave_bands(), expected)


if __name__ == '__main__':
    unittest.main()
