import sys

sys.path.append('../acoustician-tools')

import unittest
import numpy as np

from acoustician_tools.rir import *
from acoustician_tools.bands import octave_bands, third_octave_bands


class TestRIR(unittest.TestCase):
    def test_rt60_from_ir(self):
        expected = [
            2.95601022,
            2.80878383,
            2.7172609,
            2.82378924,
            2.8198684,
            2.8923384,
            2.99564819,
            2.92857127,
            2.68747812,
            2.15390832,
            1.5438748,
        ]
        calculated = rt60_from_ir('tests/IR/IR_test_big_hall.wav', octave_bands()['f_bound'], 't30')
        np.testing.assert_almost_equal(calculated, expected, decimal=1, err_msg='T30 Octave Bands - Hall IR')

        expected = [
            3.7805490287178656,
            3.1090092252819383,
            1.629469766990827,
            3.3279022066265473,
            3.0601973583418927,
            4.321770462305133,
            2.4875279403605495,
            3.252457247174818,
            4.058993073810554,
            3.967796975532446,
            3.780055451302493,
            2.4680522455699725,
            3.1536006769348286,
            3.0594573079256113,
            3.381151991861768,
            3.5692905479069323,
            3.719123620225285,
            3.3890257379800595,
            4.110989512061787,
            3.739329631799081,
            3.8974105614066845,
            3.4877162265080974,
            3.4413980713286887,
            3.324556299024433,
            3.2121337664935186,
            2.849336334438954,
            2.5845506318872973,
            2.3486788702145276,
            2.1060020345243853,
            1.7037621244440437,
            1.604616185238701,
            1.4595115467537219,
        ]
        calculated = rt60_from_ir('tests/IR/IR_test_big_hall.wav', third_octave_bands()['f_bound'], 't10')
        np.testing.assert_almost_equal(calculated, expected, decimal=1, err_msg='T10 Third Octave Bands - Hall IR')

    def test_clarity_from_ir(self):
        expected = [
            -21.794886,
            -17.234637,
            0.004106,
            -3.016408,
            2.927819,
            6.245766,
            3.787986,
            7.32541,
            7.904044,
            10.280589,
            17.862312,
        ]
        calculated = clarity_from_ir('tests/IR/IR_test.wav', octave_bands()['f_bound'], 50)
        np.testing.assert_almost_equal(calculated, expected, decimal=5, err_msg='C50 Octave Bands - Room IR')

        expected = [
            -52.795893,
            -41.246751,
            -20.530222,
            -24.0526,
            -26.786117,
            -16.351867,
            -8.693615,
            -2.067768,
            -8.266876,
            -0.929836,
            0.413873,
            2.612905,
            3.957283,
            7.154656,
            12.594287,
            7.627961,
            7.663313,
            7.06051,
            6.009544,
            7.396085,
            9.193408,
            12.847548,
            9.099557,
            10.95138,
            12.281965,
            11.279989,
            12.410985,
            14.596299,
            16.751172,
            20.532772,
            24.339488,
            31.230428,
        ]
        calculated = clarity_from_ir('tests/IR/IR_test.wav', third_octave_bands()['f_bound'], 80)
        np.testing.assert_almost_equal(calculated, expected, decimal=5, err_msg='C80 Octave Bands - Hall IR')


if __name__ == '__main__':
    unittest.main()
