"""Build power spectrum from amplitude values."""


import numpy as np


def power_spectrum(amplitude_values, sampling_frequency):
    """Apply an RFFT on the amplitude values and return the real components.

    This computes the discrete Fourier Transform using the `rfft` function
    from `numpy.fft` module and compute the frequency values using the
    `rfftfreq` from the same module.

    Args:
        amplitude_values (np.ndarray):
            A numpy array with the signal values.
        sampling_frequency (int or float):
            Sampling frequency value passed in Hz.

    Returns:
        tuple:
            * `amplitude_values (numpy.ndarray)`
            * `frequency_values (numpy.ndarray)`
    """
    frequency_values = np.fft.rfftfreq(len(amplitude_values), 1 / sampling_frequency)
    amplitude_values = np.abs(np.fft.rfft(amplitude_values)) ** 2

    return amplitude_values, frequency_values
