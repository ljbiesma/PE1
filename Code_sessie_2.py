# -*- coding: utf-8 -*-
"""
@author: s4041518
"""

# --- Imports ---
import numpy as np                     
import matplotlib.pyplot as plt        
from scipy.signal import find_peaks    
from Code_Sessie_1 import MyDAQ        



class FFTAnalyzer:
    def __init__(self, sample_rate=50000, samps_per_chan=250000):
        """
        Initialize the analyzer with the same parameters
        as the MyDAQ 
        """
        self.sample_rate = sample_rate
        self.samps_per_chan = samps_per_chan

    # ---------------------------
    def compute_fft(self, signal):
        """
        Compute FFT of a time-domain signal.
        Returns the complex FFT array and the
        frequency axis (both positive and negative).
        """
        fft = np.fft.fft(signal)                          # Perform FFT
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)  # Frequency bins
        return fft, freqs

    # ---------------------------
    def plot_fft(self, fft, freqs, title="FFT"):
        """
        Plot both magnitude and phase of FFT.
        Only shows positive frequencies (symmetry).
        """
        pos_mask = freqs >= 0        #  only positive frequencies
        freqs = freqs[pos_mask]
        fft = fft[pos_mask]

        plt.figure(figsize=(12, 5))

        # Magnitude spectrum
        plt.subplot(1, 2, 1)
        plt.loglog(freqs, np.abs(fft))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"{title} – Magnitude")

        # Phase spectrum
        plt.subplot(1, 2, 2)
        plt.semilogx(freqs, np.angle(fft))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (rad)")
        plt.title(f"{title} – Phase")

        plt.tight_layout()
        plt.show()

    # ---------------------------
    def reconstruct_from_magnitude(self, fft):
        """
        Reconstruct a signal using ONLY the magnitude
        information.
        This usually gives a noisy/unnatural signal.
        """
        mag = np.abs(fft)
        recon_fft = mag * np.exp(1j * np.zeros_like(mag))  # Set phase = 0
        return np.fft.ifft(recon_fft)

    # ---------------------------
    def reconstruct_from_phase(self, fft):
        """
        Reconstruct a signal using ONLY the phase
        information (set magnitude = 1).
        Surprisingly, this often keeps recognizable
        features like speech intelligibility.
        """
        phase = np.angle(fft)
        recon_fft = np.exp(1j * phase)  # magnitude = 1
        return np.fft.ifft(recon_fft)

    # ---------------------------
    def reconstruct_original(self, fft):
        """
        Full reconstruction: simply do an inverse FFT to get back our original signa;.
        """
        return np.fft.ifft(fft)



if __name__ == "__main__":
    myDAQ = MyDAQ()   
    analyzer = FFTAnalyzer(sample_rate=myDAQ.sample_rate,
                           samps_per_chan=myDAQ.samps_per_chan)

    print("Recording signal...")
    voltages = myDAQ.read(channel="ai0")   # Read from input channel
    myDAQ.save_data("session2_voices.npz")
    times = myDAQ.getTimeData()            
    # Plot recorded waveform
    plt.figure(figsize=(12, 4))
    plt.plot(times, voltages)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Recorded Signal (Time Domain)")
    plt.show()
    
    times, voltages = myDAQ.load_data("session2_voices.npz")
    fft, freqs = analyzer.compute_fft(voltages)  # Perform FFT
    analyzer.plot_fft(fft, freqs, title="Recorded Signal")

    # Rebuild signal in three different ways:
    recon_mag = analyzer.reconstruct_from_magnitude(fft)
    recon_phase = analyzer.reconstruct_from_phase(fft)
    recon_orig = analyzer.reconstruct_original(fft)

    # Plot the three reconstructions
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(times, np.real(recon_orig))
    plt.title("Reconstruction with Full FFT (Original)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(times, np.real(recon_mag))
    plt.title("Reconstruction with Magnitude Only")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(times, np.real(recon_phase))
    plt.title("Reconstruction with Phase Only")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
    
    # find_peaks finds prominent frequencies in the FFT spectrum
    peaks, _ = find_peaks(np.abs(fft), height=0.1)
    peak_freqs = freqs[peaks]
    print("Detected frequency peaks (Hz):", peak_freqs)
    response = myDAQ.write(recon_phase)
