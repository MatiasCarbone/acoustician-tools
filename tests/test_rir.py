import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np

from acoustician_tools.rir import *


class TestRIR(unittest.TestCase):
    def test_load_ir_mono(self):
        length = 262144
        first_10 = [
            9.687375e-06,
            -5.1842585e-06,
            -3.502343e-05,
            -2.176988e-05,
            2.126425e-06,
            -3.8547214e-06,
            -1.7837274e-05,
            -5.584359e-06,
            1.5538884e-06,
            -3.911781e-06,
        ]
        ir = load_ir_mono('tests/IR/IR_test.txt')
        self.assertEqual(len(ir), length)
        self.assertEqual(ir[0:10], first_10)


if __name__ == '__main__':
    unittest.main()
