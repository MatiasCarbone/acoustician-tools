import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np
from acoustician_tools.utils import *


class TestUtils(unittest.TestCase):
    def test_sound_speed(self):
        expected = 343.21
        self.assertAlmostEqual(sound_speed(20.0), expected, places=1)
