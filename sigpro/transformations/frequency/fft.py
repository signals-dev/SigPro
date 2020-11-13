"""SigPro Transformations Frequency FFT module."""

import numpy as np


def fft_real(amplitude_values, sampling_frequency):
    """Apply an FFT on the amplitude values and return the real components.

    This computes the discrete Fourier Transform using the `fft` function
    from `numpy.fft` module and then extracting the real part of the
    returned vector to discard the complex components that represent
    the phase values.
    Also compute the frequency values using the `fftfreq` from the
    same module.

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
    amplitude_values = np.real(np.fft.fft(amplitude_values))
    frequency_values = np.real(np.fft.fftfreq(len(amplitude_values), sampling_frequency))

    return amplitude_values, frequency_values
