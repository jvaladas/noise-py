# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:37:40 2019

@author: NB22909
"""

import sys
import os
from os.path import isfile, join, dirname

directory = dirname(__file__)
sys.path.append(directory)

import subprocess
from joblib import Parallel, delayed
import multiprocessing

import scipy.io.wavfile as wav
import noise_reduction
from scipy import signal, stats
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import pyfftw as fftw
from math import e

def plot_graphs(file):
    print "\nPlotting file: " + str(file)

    y, sr = librosa.load(file, sr=None)
    stft = librosa.stft(y)
    S = np.abs(stft)

    contrast = librosa.feature.spectral_contrast(S = S, sr = sr, fmin = 100 )
    flatness = librosa.feature.spectral_flatness(y = y)
    rms = np.mean(stats.trim_mean(librosa.feature.rms(S = S), 0.1)) * 100
    rolloff = np.mean(librosa.feature.spectral_rolloff(S = S))
    bandwidth = librosa.feature.spectral_bandwidth(S = S, sr = sr)
    centroid = librosa.feature.spectral_centroid(S = S, sr = sr)

    print('sample rate: ', sr)
    print('mean contrast: ', np.mean(stats.trim_mean(contrast, 0.1)))
    print('spec flatness: ', np.mean(stats.trim_mean(flatness, 0.1)))
    print('rms: ', rms)
    print('centroid: ', np.mean(stats.trim_mean(centroid, 0.1)))
    print('rolloff: ', rolloff)
    print('bandwidth: ', np.mean(stats.trim_mean(bandwidth, 0.1)))

    fourier = fftw.interfaces.numpy_fft.fft(y)

    #draw before graph
    max_len = np.floor(len(fourier)/2 -1).astype('int')
    plt.plot(abs(fourier[:(max_len-1)]),'r')
    plt.show()


    max_value = np.argmax(abs(fourier[1:len(fourier/2)-1]))
    frequencies = np.fft.fftfreq(len(fourier/2))
    max_freq = abs(frequencies[int(max_value)] * sr)

    print('max freq: ', max_freq)

    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref = np.max(S)), y_axis = 'log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectogram')
    plt.tight_layout()
    plt.show()

    return rolloff, rms


def reduce_noise(file):

    print("Reducing noise on file: " + file)
    (rate, sample) = wav.read(file)

    spec_rolloff, rms = plot_graphs(file)

    nr = noise_reduction.NoiseRemover()
    output = nr.remove_noise(sample, rate, spec_rolloff, rms)

    file = file.replace('\\', '/')
    parts = file.split("/")
    filename = parts[len(parts) - 1]

    path = join(directory, 'filtered_' + filename)

    wav.write(path, rate, output.astype('int16'))
    return path


def process_file(folder, current_file):

    wavesplitter = join(directory, 'WaveSplitter/wsplitter.exe')
    sox = join(directory, 'Sox/sox.exe')

    print("Splitting file: " + current_file)
    subprocess.check_call([wavesplitter, join(folder, current_file)])

    # process each result
    left_channel = join(directory, 'input_data/L_' + current_file.replace(".mp3", ".wav"))
    right_channel = join(directory, 'input_data/R_' + current_file.replace(".mp3", ".wav"))

    filter_left = reduce_noise(left_channel)
    filter_right = reduce_noise(right_channel)

    result_path = join(directory, 'results/improved_' + current_file.replace(".mp3", ".wav"))

    # join resulting tracks with sox
    subprocess.check_call([sox, "-M", "-c", "1", filter_left, "-c", "1", filter_right, result_path])

    # delete temp l and r
    print("Removing temporary channel and filtered channel files.")
    os.remove(left_channel)
    os.remove(right_channel)
    os.remove(filter_left)
    os.remove(filter_right)



def parallel_reduction(folder):

    # sets up parallel for
    num_cores = multiprocessing.cpu_count()
    print("The system has " + str(num_cores) + " available cores.")

    collection = [f for f in os.listdir(folder) if isfile(join(folder, f))]

    if num_cores > len(collection):
        num_cores = len(collection)

    Parallel(n_jobs = num_cores, verbose = 100)(delayed(process_file)(folder, current_file) for current_file in collection)


def sequential_reduction(folder):
    collection = [f for f in os.listdir(folder) if isfile(join(folder,f))]
    for current_file in collection:
        process_file(folder, current_file)


#parallel_reduction(join(directory, 'input_data'))

#reduce_noise(join(directory, 'VerifyErrado.wav'))
#reduce_noise(join(directory, 'VerifyCorreto.wav'))

#plot_graphs(join(directory, 'VerifyErrado.wav'))
#plot_graphs(join(directory, 'VerifyCorreto.wav'))

#plot_graphs(join(directory, 'filtered_VerifyCorreto.wav'))
#plot_graphs(join(directory, 'filtered_VerifyErrado.wav'))

#plot_graphs(join(directory, '20190529_165957_50081_999995022_f0702f8d77c54b9484ce.wav'))
#plot_graphs(join(directory, 'improved_20190529_165957_50081_999995022_f0702f8d77c54b9484ce.wav'))

#plot_graphs(join(directory, 'ola.wav'))
#plot_graphs(join(directory, 'improved_ola.wav'))

#plot_graphs(join(directory, '440hz.wav'))
#plot_graphs(join(directory, 'whitenoise.wav'))

sequential_reduction(join(directory, 'input_data'))



