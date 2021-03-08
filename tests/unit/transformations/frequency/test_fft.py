"""Tests for sigpro.transformations.frequency.fft module."""
import numpy as np

from sigpro.transformations.frequency.fft import fft, fft_real


def test_fft():
    # setup
    values = [1, 1, 1, 1, 1]

    # run
    amplitude_values, frequency_values = fft(values, 10)

    # assert
    expected_amplitude_values = [5. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]
    expected_frequency_values = [0., 2., 4., -4., -2.]
    np.testing.assert_array_almost_equal(amplitude_values, expected_amplitude_values)
    np.testing.assert_array_almost_equal(frequency_values, expected_frequency_values)


def test_fft_real():
    # setup
    values = [1, 1, 0, 1, 1]

    # run
    amplitude_values, frequency_values = fft_real(values, 10)

    # assert
    expected_amplitude_values = [4.0, 0.80901699, -0.309017, -0.309017, 0.809017]
    expected_frequency_values = [0., 2., 4., -4., -2.]
    np.testing.assert_array_almost_equal(amplitude_values, expected_amplitude_values)
    np.testing.assert_array_almost_equal(frequency_values, expected_frequency_values)
