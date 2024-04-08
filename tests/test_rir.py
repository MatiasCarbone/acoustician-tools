import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np

from acoustician_tools.rir import *


class TestRIR(unittest.TestCase):
    # def test_rt60_from_ir(self):
    #     expected = 0.512
    #     calculated = rt60_from_ir('tests/IR/IR_test.wav', [(88, 177)], 't30')

    #     self.assertAlmostEqual(*calculated, expected)


if __name__ == '__main__':
    unittest.main()
