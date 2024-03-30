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
            msg='Sound speed at 20.0 °C is approximately 343 m/s',
        )

    def test_shoebox_surfaces(self):
        expected = [8, 8, 6, 6, 12, 12]
        self.assertEqual(
            shoebox_surfaces(4, 3, 2),
            expected,
            msg='The boundary-surfaces for a 4x3x2 room should be 8-8-6-6-12-12',
        )
