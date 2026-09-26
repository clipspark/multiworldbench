from __future__ import annotations

import unittest

import numpy as np

from controller.multiworldbench_v3.engines.rotation import (
    interpolate_rotations,
    rotation_magnitude,
)


class RotationTests(unittest.TestCase):
    def test_magnitude_and_slerp_midpoint(self):
        identity = np.eye(3)
        quarter_turn = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        magnitudes = rotation_magnitude(np.stack([identity, quarter_turn]))
        np.testing.assert_allclose(magnitudes, [0.0, np.pi / 2.0], atol=1e-8)

        midpoint = interpolate_rotations(
            np.array([0.0, 1.0]),
            np.stack([identity, quarter_turn]),
            np.array([0.5]),
        )[0]
        expected = np.sqrt(0.5)
        np.testing.assert_allclose(
            midpoint,
            [[expected, -expected, 0.0],
             [expected, expected, 0.0],
             [0.0, 0.0, 1.0]],
            atol=1e-8,
        )

    def test_constant_rotation_and_input_validation(self):
        identity = np.eye(3)
        result = interpolate_rotations(
            np.array([3.0]), np.stack([identity]), np.array([1.0, 5.0])
        )
        np.testing.assert_allclose(result, np.stack([identity, identity]))
        with self.assertRaises(ValueError):
            interpolate_rotations(
                np.array([0.0, 0.0]),
                np.stack([identity, identity]),
                np.array([0.0]),
            )


if __name__ == "__main__":
    unittest.main()
