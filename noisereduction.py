# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:01:34 2020

@author: jude.dinoso
"""

import noisereduce as nr
import numpy as np

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real

def noisereduction(sound_array, samplingrate, seconds):
    sound_array = sound_array.reshape(samplingrate*seconds)
    noisy_part = sound_array[000:10000]
    clean_audio = nr.reduce_noise(audio_clip = sound_array, noise_clip=noisy_part, verbose=False) * 32768
    export_audio =clean_audio.astype(np.int16, order='C')
    return export_audio

    