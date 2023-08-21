# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:31:22 2013
@author: Lukasz Tracewski
Various filters
"""

from scipy.signal import firwin, butter, lfilter, kaiserord, wiener
import scipy.signal as sig
from scipy.stats import trim_mean
import numpy as np
from segmentation import Segmentator
from noise_subtraction import reduce_noise
from utilities import contiguous_regions
import librosa
import time
import pyfftw as fftw

class NoiseRemover(object):

    def remove_noise(self, signal, rate, spec_rolloff=0.0, rms = 15.0):

        signal = remove_clicks(signal, rate, window_size=2**10, margin=1.2)

        fourier = fftw.interfaces.numpy_fft.fft(signal)
        n = fourier.size/2
        frequencies = np.fft.fftfreq(n)
        max_value = np.argmax(abs(fourier[1:n-1]))
        max_freq = abs(frequencies[int(max_value)] * rate)

        # improve signal strength
        for i in range(0, fourier.size - 1):
            if(rms > 100):
                fourier[i] = fourier[i] / (rms/300)
            elif(rms < 10):
                fourier[i] = fourier[i] * 3
            else:
                fourier[i] = fourier[i] * 2.7

        signal = fftw.interfaces.numpy_fft.ifft(fourier)

        # decide low and high cuts on frequencies
        max_freq = abs(frequencies[int(max_value)] * rate)
        low_cut = 70
        high_cut = max_freq + 3500

        if(max_freq > 1250 + low_cut):
            low_cut = max_freq - 1500

        if(high_cut >= rate / 2):
            high_cut = rate / 2 - 1

        print "low_cut: " + str(low_cut) + " high_cut: " + str(high_cut)

        self.sample_highpassed = signal
        self.segmentator = select_best_segmentator(self.sample_highpassed, rate, 'mkl')

        no_silence_intervals = self.segmentator.get_number_of_silence_intervals()
        print('number of silence intervals: ', no_silence_intervals )
        out = signal

        if no_silence_intervals == 0:
            self.sample_highpassed = bandpass_filter(self.sample_highpassed, rate, low_cut, high_cut)
            return self.sample_highpassed

        elif no_silence_intervals == 1:
            noise = self.segmentator.get_next_silence(signal)  # Get silence period
            out = reduce_noise(signal, noise)  # Perform spectral subtraction
        else:
            max_iterations = 3
            for i in range(0, no_silence_intervals):
                if max_iterations == i:
                    break

                print "reducing interval " + str(i)
                noise = self.segmentator.get_next_silence(signal)  # Get silence period
                out = reduce_noise(out, noise)  # Perform spectral subtraction

        # Apply high-pass filter on spectral-subtracted sample
        out = bandpass_filter(out, rate, low_cut, high_cut)
        return out


def select_best_segmentator(signal, rate, detector):
    # Segmentator divides a track into segments containing sound features (non-noise)
    # and silence (noise)
    segmentator = Segmentator(detector_type=detector, threshold=0.01)

    # Perform segmentation on high-passed sample
    segmentator.process(signal, rate)

    no_silence_intervals = segmentator.get_number_of_silence_intervals()

    if no_silence_intervals < 2:
        segmentator = Segmentator(detector_type=detector, threshold=0.2)
        segmentator.process(signal, rate)

    return segmentator


def wiener_filter(signal):
    """ Wiener filter. """
    output = wiener(signal, 2**5 - 1)
    return output


def remove_clicks(signal, rate, window_size, margin):
    """
    Clicks are bursts of energy. The fucntion will calculate signal energy over given window
    size and eliminate regions of abnormaly high energy content.
    Parameters
    --------------
    signal : 1d-array
        Single-channel audio sample.
    rate : int
        Sample rate in Hz.
    window_size : int
        The number of data points used in each block for the energy calculation.
    margin : float
        How much (in seconds) the sample should be cut on both sides.
    Returns
    --------------
    signal : 1d-array
        Cleared signal.
    """
    margin = margin * rate
    overlap = window_size / 2.0
    mask = np.ones(len(signal), dtype=bool)

    if np.abs(signal.max()) > 2**14:
        energy = calculate_energy(signal, window_size, overlap)
        energy = sig.medfilt(energy, 15)

        p = np.percentile(energy, 90)
        condition = energy > 2*p

        cont = contiguous_regions(condition)
        for start, stop in cont:
            start_idx = int(start * window_size - margin)
            stop_idx = int(stop * window_size + margin)
            mask[start_idx:stop_idx] = False

    return signal[mask]


def calculate_energy(signal, period, overlap=0):
    """ Calculate energy of the signal. """

    half_overlap = int(overlap/2)
    intervals = np.arange(0, len(signal), period)
    energy = np.zeros(len(intervals) - 1)

    energy_slice_start = signal[intervals[0]:intervals[1] + half_overlap]
    energy[0] = sum(energy_slice_start**2)

    energy_slice_end = signal[intervals[-2] - half_overlap:intervals[-1]]
    energy[-1] = sum(energy_slice_end**2)

    for i in np.arange(1, len(intervals) - 2):
        energy_slice = signal[intervals[i] - half_overlap:intervals[i+1] + half_overlap]
        energy[i] = sum(energy_slice**2)

    return energy


def highpass_filter(signal, rate, cut):
    """ Apply highpass filter. Everything below 'cut' frequency level will
        be greatly reduced. """
    ntaps = 199
    nyq = 0.5 * rate
    highpass = firwin(ntaps, cut, nyq=nyq, pass_zero=False,
                      window='hann', scale=False)
    filteredSignal = lfilter(highpass, 1.0, signal)
    return filteredSignal


def bandpass_filter(signal, signal_rate, lowcut, highcut, window='hann'):
    """ Apply bandpass filter. Everything below 'lowcut' and above 'highcut'
        frequency level will be greatly reduced. """
    ntaps = 199
    nyq = 0.5 * signal_rate
    lowpass = firwin(ntaps, lowcut, nyq=nyq, pass_zero=False,
                     window=window, scale=False)
    highpass = - firwin(ntaps, highcut, nyq=nyq, pass_zero=False,
                        window=window, scale=False)
    highpass[ntaps/2] = highpass[ntaps/2] + 1
    bandpass = -(lowpass + highpass)
    bandpass[ntaps/2] = bandpass[ntaps/2] + 1
    filteredSignal = lfilter(bandpass, 1.0, signal)
    return filteredSignal


def keiser_bandpass_filter(signal, signal_rate, lowcut, highcut):
    """ Apply bandpass filter. Everything below 'lowcut' and above 'highcut'
        frequency level will be greatly reduced. Keiser edition. """
    # The Nyquist rate of the signal.
    nyq_rate = 0.5 * signal_rate

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0 / nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    lowpass = firwin(N, lowcut / nyq_rate, window=('kaiser', beta))
    highpass = - firwin(N, highcut / nyq_rate, window=('kaiser', beta))
    highpass[N/2] = highpass[N/2] + 1
    bandpass = - (lowpass + highpass)
    bandpass[N/2] = bandpass[N/2] + 1
    # Use lfilter to filter x with the FIR filter.
    filteredSignal = lfilter(bandpass, 1.0, signal)
    return filteredSignal


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ Apply _bandpass filter. Everything below 'lowcut' and above 'highcut'
        frequency level will be greatly reduced. Butter edition. """
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def moving_average(signal, length):
    """ Moving average filter. """
    smoothed = np.convolve(signal, np.ones(length)/length, mode='same')
    return smoothed
