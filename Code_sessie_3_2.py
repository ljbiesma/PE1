# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:52:12 2025

@author: ljbie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Code_Sessie_1 import MyDAQ

def amplitude_phase(signal, freq, fs):
    """Extract amplitude and phase at given frequency using correlation."""
    t = np.arange(len(signal)) / fs
    ref_cos = np.cos(2*np.pi*freq*t)
    ref_sin = np.sin(2*np.pi*freq*t)
    a = 2*np.mean(signal * ref_cos)
    b = 2*np.mean(signal * ref_sin)
    amp = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)
    return amp, phase

def measure_single_frequency(myDAQ, freq, amplitude=1.0, repeats=2):
    """Measure amplitude ratio |H| at one frequency."""
    H_vals = []
    fs = myDAQ.sample_rate
    for i in range(repeats):
        duration = 20 / freq   # capture 20 cycles
        n = int(fs * duration)
        t = np.arange(n) / fs
        sine = myDAQ.gen_sine(frequency=freq, amplitude=amplitude)
        resp = myDAQ.write_and_read(sine[:n])
        Ain, Aout = amplitude_phase(sine[:n], freq, fs), amplitude_phase(resp[:n], freq, fs)
        H_vals.append(Aout[0] / Ain[0])
    return np.mean(H_vals), np.std(H_vals)

def measure_sweep(myDAQ, freqs, amplitude=1.0):
    H_mean, H_std = [], []
    for f in freqs:
        print(f"Measuring {f:.1f} Hz")
        m, s = measure_single_frequency(myDAQ, f, amplitude)
        H_mean.append(m)
        H_std.append(s)
    return np.array(freqs), np.array(H_mean), np.array(H_std)

def fit_cutoff(freqs, H_means, H_stds):
    def H_model(f, f_c): return 1 / np.sqrt(1 + (f/f_c)**2)
    popt, pcov = curve_fit(H_model, freqs, H_means, sigma=H_stds, p0=[1000], maxfev=5000)
    f_c = popt[0]; f_c_err = np.sqrt(np.diag(pcov))[0]
    plt.figure(figsize=(8,5))
    plt.errorbar(freqs, 20*np.log10(H_means), yerr=(20/np.log(10))*H_stds/H_means,
                 fmt='o', label='Data', capsize=3)
    f_fit = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 400)
    plt.semilogx(f_fit, 20*np.log10(H_model(f_fit, f_c)), '-', label=f'Fit fc={f_c:.1f} Hz')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|H| (dB)")
    plt.title("RC Magnitude Transfer Function")
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout(); plt.show()
    print(f"Cutoff frequency fc = {f_c:.2f} ± {f_c_err:.2f} Hz")
    return f_c, f_c_err

if __name__ == "__main__":
    myDAQ = MyDAQ()
    freqs = np.logspace(1, 4.3, 25)  # 10–20 000 Hz
    freqs, H_mean, H_std = measure_sweep(myDAQ, freqs, amplitude=1.0)
    fit_cutoff(freqs, H_mean, H_std)
