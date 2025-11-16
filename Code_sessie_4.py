# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 13:36:43 2025

@author: ljbie
"""

# -*- coding: utf-8 -*-
"""
Session 4 – The RC Transfer Function
Measure both magnitude and phase of an RC filter’s transfer function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Code_Sessie_1 import MyDAQ
from Code_sessie_2 import FFTAnalyzer


# ----------------------------------------------------------------------
# Helper: amplitude and phase extraction (lock-in style)
# ----------------------------------------------------------------------
"""
def amplitude_phase(signal, freq, fs):
    Return amplitude and phase (rad) of signal at given frequency.
    t = np.arange(len(signal)) / fs
    ref_cos = np.cos(2 * np.pi * freq * t)
    ref_sin = np.sin(2 * np.pi * freq * t)
    a = 2 * np.mean(signal * ref_cos)
    b = 2 * np.mean(signal * ref_sin)
    amp = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)
    return amp, phase
"""

# ----------------------------------------------------------------------
# Measure single frequency
# ----------------------------------------------------------------------
def measure_single_frequency(myDAQ, analyzer, freq, amplitude=1.0, repeats=3):
    """Return mean |H| and phase (rad) for one frequency."""
    fs = myDAQ.sample_rate
    H_vals, ph_vals = [], []

    for i in range(repeats):
        # Generate sine wave and measure response
        sine = myDAQ.gen_sine(frequency=freq, amplitude=amplitude)
        response = myDAQ.write_and_read(sine, ao_channel="ao0", ai_channels_str="ai0, ai1")

        Uin = response[1]
        Uout = response[0]

        # --- Compute FFTs ---
        fft_in, f_axis = analyzer.compute_fft(Uin)
        fft_out, _ = analyzer.compute_fft(Uout)
        idx = np.argmin(np.abs(f_axis - freq))
        H_vals.append(np.abs(fft_out[idx]) / np.abs(fft_in[idx]))
        ph_vals.append(np.mod(np.angle(fft_out[idx]) / (np.angle(fft_in[idx])),360))

        """
        duration = 20 / freq              # capture ~20 cycles
        n = int(fs * duration)
        t = np.arange(n) / fs
        sine = amplitude * np.sin(2 * np.pi * freq * t)
        resp = myDAQ.write_and_read(sine[:n])

        A_in, ph_in = amplitude_phase(sine[:n], freq, fs)
        A_out, ph_out = amplitude_phase(resp[:n], freq, fs)

        H_vals.append(A_out / A_in)
        ph_vals.append(ph_out - ph_in)    # phase difference
        """
    H_vals = np.array(H_vals)
    ph_vals = np.unwrap(np.array(ph_vals))  # unwrap avoids 360° jumps
    return np.mean(H_vals), np.std(H_vals), np.mean(ph_vals)


# ----------------------------------------------------------------------
# Sweep frequencies
# ----------------------------------------------------------------------
def measure_sweep(myDAQ, analyzer, freqs, amplitude=1.0, repeats=3):
    """Run sweep over frequencies, returning |H| and phase arrays."""
    H_mean, H_std, PH_mean = [], [], []
    for f in freqs:
        print(f"Measuring {f:.1f} Hz")
        m, s, p = measure_single_frequency(myDAQ, analyzer, f, amplitude, repeats)
        H_mean.append(m)
        H_std.append(s)
        PH_mean.append(p)
    return np.array(freqs), np.array(H_mean), np.array(H_std), np.array(PH_mean)


# ----------------------------------------------------------------------
# Fit and plot Bode
# ----------------------------------------------------------------------
def fit_and_plot(freqs, H_means, H_stds, phases):
    """Fit magnitude to RC model and plot full Bode diagram."""
    def H_model(f, f_c):  # first-order low-pass magnitude
        return 1 / np.sqrt(1 + (f / f_c)**2)

    # --- Fit magnitude data ---
    popt, pcov = curve_fit(H_model, freqs, H_means, sigma=H_stds, p0=[1000])
    f_c = popt[0]
    f_c_err = np.sqrt(np.diag(pcov))[0]

    # --- Prepare theoretical curves ---
    f_fit = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 400)
    H_fit = H_model(f_fit, f_c)
    phase_fit = -np.arctan(f_fit / f_c) * 180 / np.pi  # theory in degrees

    # --- Plot magnitude (dB) ---
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.errorbar(freqs, 20 * np.log10(H_means),
                 yerr=(20 / np.log(10)) * H_stds / H_means,
                 fmt="o", label="Measurements", capsize=3)
    plt.semilogx(f_fit, 20 * np.log10(H_fit), "-", label=f"Fit fc={f_c:.1f} Hz")
    plt.ylabel("Magnitude |H| (dB)")
    plt.title("RC Transfer Function – Magnitude and Phase")
    plt.grid(True, which="both")
    plt.legend()

    # --- Plot phase (deg) ---
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs, np.degrees(phases), "o", label="Measured")
    plt.semilogx(f_fit, phase_fit, "-", label="Theoretical")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (°)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Cutoff frequency fc = {f_c:.2f} ± {f_c_err:.2f} Hz")
    return f_c, f_c_err


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
if __name__ == "__main__":
    myDAQ = MyDAQ()
    #analyzer = FFTAnalyzer(
        #sample_rate=myDAQ.sample_rate, samps_per_chan=myDAQ.samps_per_chan
    #)
    #freqs = np.logspace(1, 4.3, 30)  # 10 Hz → 20 kHz
    #freqs, H_mean, H_std, PH_mean = measure_sweep(
        #myDAQ, analyzer, freqs=freqs, amplitude=1.0, repeats=3)
    #fit_and_plot(freqs, H_mean, H_std, PH_mean)
    bestandsnaam = "mijn_laatste_meting.npz" 
    myDAQ.save_data(filename=bestandsnaam)
