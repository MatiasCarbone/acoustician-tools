import sys

sys.path.append('../acoustician-tools')

import unittest
from acoustician_tools.room import t60_sabine


class TestRoom(unittest.TestCase):
    def setUp(self):
        self.volume = 3000
        self.surfaces = [240, 600, 500]
        self.alphas = [0.1, 0.25, 0.45]

    def test_sabine(self):
        expected = 1.2113
        self.assertAlmostEqual(
            t60_sabine(self.volume, self.surfaces, self.alphas),
            expected,
            msg=f'Should be close to {expected}.',
            places=2,
        )


if __name__ == '__main__':
    unittest.main()
