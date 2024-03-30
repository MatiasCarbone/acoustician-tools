import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np

from acoustician_tools.absorber import *


class TestAbsorber(unittest.TestCase):
    def test_nrc(self):
        alphas = [0.29, 0.47, 0.79, 0.92]
        expected = 0.6
        self.assertEqual(nrc(alphas), expected, msg='Normal alpha values for NRC')

        alphas = [1.2, 0.9, 1.1, 0.8]
        expected = 1.0
        self.assertEqual(nrc(alphas), expected, msg='NRC more than 1.0')

        alphas = [-1.0, 0.1, 0.3, 0.1]
        with self.assertRaises(Exception, msg='Negative alpha coefficient'):
            nrc(alphas)

        alphas = [-1.0, 0.1, 'test']
        with self.assertRaises(Exception, msg='Incorrect list length'):
            nrc(alphas)

    def test_porous_absorber(self):
        expected = [0.13, 0.49, 0.95, 0.96, 0.99]
        calculated = porous_absorber(
            flow_resistivity=10100,
            thickness=100,
            frequencies=[100, 200, 500, 1000, 2000],
        )
        np.testing.assert_almost_equal(expected, calculated, decimal=2)

        expected = [0.04, 0.17, 0.4, 0.65, 0.77, 0.88, 0.96]
        calculated = porous_absorber(
            flow_resistivity=45000,
            thickness=71,
            frequencies=[50, 100, 200, 500, 1000, 2000, 4000],
        )
        np.testing.assert_almost_equal(expected, calculated, decimal=2)


if __name__ == '__main__':
    unittest.main()
