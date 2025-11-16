# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:24:24 2025

@author: ljbie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Code_Sessie_1 import MyDAQ
from Code_sessie_2 import FFTAnalyzer



def measure_single_frequency(myDAQ, analyzer, frequency=500, amplitude=1.0, repeats=3):
    H_values = []

    for i in range(repeats):
        print(f"  Repetition {i+1}/{repeats} at {frequency:.1f} Hz")

        # Generate sine wave and measure response
        sine = myDAQ.gen_sine(frequency=frequency, amplitude=amplitude)
        response = myDAQ.write_and_read(sine, ao_channel="ao0", ai_channels_str="ai0")

        Uin = sine
        Uout = response

        # --- Compute FFTs ---
        fft_in, f_axis = analyzer.compute_fft(Uin)
        fft_out, _ = analyzer.compute_fft(Uout)
        idx = np.argmin(np.abs(f_axis - frequency))
        H_values.append(np.abs(fft_out[idx]) / np.abs(fft_in[idx]))

        # Time axis for plotting
        t = myDAQ.getTimeData()
        """
        # Plot (only for first repetition, to save time)
        if i == 0:
            # --- TIME DOMAIN ---
            plt.figure(figsize=(9,4))
            plt.plot(t, sine, label="Input (U_in)")
            plt.plot(t, response, label="Output (U_out)")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.title(f"RC Filter Response at {frequency:.0f} Hz (Time Domain)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # --- FREQUENCY DOMAIN ---
            pos_mask = f_axis >= 0
            plt.figure(figsize=(9,4))
            plt.semilogx(f_axis[pos_mask],
                         20*np.log10(np.abs(fft_in[pos_mask])/np.max(np.abs(fft_in))),
                         label="Input Spectrum")
            plt.semilogx(f_axis[pos_mask],
                         20*np.log10(np.abs(fft_out[pos_mask])/np.max(np.abs(fft_out))),
                         label="Output Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB, normalized)")
            plt.title(f"RC Filter Response at {frequency:.0f} Hz (Frequency Domain)")
            plt.legend()
            plt.grid(True, which="both")
            plt.tight_layout()
            plt.show()
            
        """

    # Calculate mean and std deviation for |H|
    H_values = np.array(H_values)
    H_mean = np.mean(H_values)
    H_std = np.std(H_values)

    print(f"|H| = {H_mean:.3f} ± {H_std:.3f} at {frequency:.1f} Hz\n")
    return H_mean, H_std


def measure_sweep(myDAQ, analyzer, freqs, amplitude=1.0, repeats=3):
    H_means = []
    H_stds = []

    # --- Loop over all frequencies ---
    for f in freqs:
        print(f"Measuring {f:.1f} Hz")
        H_mean, H_std = measure_single_frequency(
            myDAQ, analyzer, frequency=f, amplitude=amplitude, repeats=repeats
        )
        H_means.append(H_mean)
        H_stds.append(H_std)

    H_means = np.array(H_means)
    H_stds = np.array(H_stds)

    # --- Save data to file ---
    np.savez("Session3_sweep_data.npz", freqs=freqs, Hmag=H_means, Hstd=H_stds)
    print("Sweep data saved to Session3_sweep_data.npz")

    return freqs, H_means, H_stds


def fit_cutoff_frequency(freqs, H_means, H_stds):
    # --- Theoretical model for a first-order low-pass RC filter ---
    def H_model(f, RC):
        return 1.0 / np.sqrt(1 + (2 * np.pi * f * RC) ** 2)

    # --- Fit the data (weighted by measurement uncertainty) ---
    popt, pcov = curve_fit(
        H_model, freqs, H_means, sigma=H_stds, absolute_sigma=True
    )

    # Extract fit parameters and uncertainties
    RC_fit = popt[0]
    RC_err = np.sqrt(np.diag(pcov))[0]
    f_c = 1.0 / (2 * np.pi * RC_fit)
    f_c_err = RC_err / (2 * np.pi * RC_fit ** 2)

    print(f"Fit results:")
    print(f"  RC = {RC_fit:.3e} ± {RC_err:.3e}  [s]")
    print(f"  f_c = {f_c:.1f} ± {f_c_err:.1f}  [Hz]")

    # --- Generate smooth theoretical curve for plotting ---
    f_fit = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 100)
    H_fit = H_model(f_fit, RC_fit)

    # --- Plot with error bars and fitted curve ---
    plt.figure(figsize=(8,5))
    plt.errorbar(
        freqs,
        20*np.log10(H_means),
        yerr=(20/np.log(10))*H_stds/H_means,  # convert to dB
        fmt="o",
        label="Measurements",
        capsize=3,
    )
    plt.semilogx(f_fit, 20*np.log10(H_fit), "-", label="Fit")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude |H| (dB)")
    plt.title("RC Filter Magnitude Transfer Function")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return f_c, f_c_err



if __name__ == "__main__":

    # Initialize MyDAQ and FFT analyzer using same sampling settings
    myDAQ = MyDAQ()
    analyzer = FFTAnalyzer(
        sample_rate=myDAQ.sample_rate, samps_per_chan=myDAQ.samps_per_chan
    )

    print("RC Transfer Function Measurement")
    mode = input("Run single frequency (s) or full sweep (a)? [s/a]: ").strip().lower()

    if mode == "s":
        # --- SINGLE-FREQUENCY MODE ---
        # Quick test to verify correct setup and data flow
        freq = float(input("Enter frequency in Hz (e.g. 1000): ") or 1000)
        measure_single_frequency(myDAQ, analyzer, frequency=freq)

    elif mode == "a":
        # --- AUTOMATED SWEEP MODE ---
        print("Running automated logarithmic sweep 10 Hz → 20 kHz")

        # Logarithmic spacing gives denser sampling near f_c
        freqs = np.logspace(1, np.log10(20000), 100)

        # Run sweep with repetitions to estimate uncertainty
        freqs, H_means, H_stds = measure_sweep(
            myDAQ, analyzer, freqs=freqs, amplitude=1.0, repeats=3
        )

        # Fit RC model and compute cut-off frequency with uncertainty
        f_c, f_c_err = fit_cutoff_frequency(freqs, H_means, H_stds)
        print(f"\nEstimated cut-off frequency: f_c = {f_c:.1f} ± {f_c_err:.1f} Hz")