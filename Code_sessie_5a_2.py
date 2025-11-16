# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 14:29:00 2025

@author: ljbie
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:36:42 2025

@author: ljbie
"""

import nidaqmx as dx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks    
import numpy as np
import time
from scipy.optimize import curve_fit
from scipy.stats import linregress





class MyDAQ():
    def __init__(self):
        self.device = "myDAQ1"
        self.sample_rate = 50000
        self.samps_per_chan = 55000
        
    def write(self, data, channel="ao0"): #writes an array of voltages
        with dx.Task() as writeTask:
            writeTask.ao_channels.add_ao_voltage_chan(f"{self.device}/{channel}")
            writeTask.timing.cfg_samp_clk_timing(
                self.sample_rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(data),
            )
            writeTask.write(data, auto_start=True)
            time.sleep(len(data) / self.sample_rate + 0.01) 
        
    def gen_sine(self, frequency, amplitude=1.0, phase=0.0, offset=0.0): #generates a sine array
        t = np.arange(self.samps_per_chan) / self.sample_rate
        sine = offset + amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return sine

    def read(self, channel="ai0"):#reads voltages and stores last data
        with dx.Task() as readTask:
            readTask.ai_channels.add_ai_voltage_chan(f"{self.device}/{channel}")
            readTask.timing.cfg_samp_clk_timing(
                self.sample_rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=self.samps_per_chan,
            )
            data = readTask.read(number_of_samples_per_channel=self.samps_per_chan)
            self._last_data = np.array(data)
            self._last_time = np.arange(self.samps_per_chan) / self.sample_rate
        return self._last_data
    def getVoltData(self): #used return the previous data
        return self._last_data
    def getTimeData(self): #Return the time array corresponding to the last recorded data.
        return self._last_time 

    def write_and_read(self, out_data, ao_channel="ao0", ai_channels_str="ai0"): #Simultaneously writes to an output channel and read from an input channel.
        with dx.Task("AOTask") as writeTask, dx.Task("AITask") as readTask:
            # Setup AO
            writeTask.ao_channels.add_ao_voltage_chan(f"{self.device}/{ao_channel}")
            writeTask.timing.cfg_samp_clk_timing(
                self.sample_rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(out_data),
            )

            # Setup AI
            channels = ai_channels_str.split(',')
            full_channel_string = ", ".join([f"{self.device}/{ch.strip()}" for ch in channels])

            readTask.ai_channels.add_ai_voltage_chan(full_channel_string)
            readTask.timing.cfg_samp_clk_timing(
                self.sample_rate,
                sample_mode=dx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(out_data),
            )

            # Run tasks
            writeTask.write(out_data, auto_start=False)
            writeTask.start()
            data = readTask.read(number_of_samples_per_channel=len(out_data))
            time.sleep(len(out_data) / self.sample_rate + 0.01)

        self._last_data = np.array(data)
        self._last_time = np.arange(len(out_data)) / self.sample_rate
        return self._last_data
    def save_data(self, filename="sweep_raw_data.npz",
              freqs=None, Uin_all=None, Uout_all=None, t=None):
        """
        Save full sweep data.
        """
        np.savez(filename,
             freqs=freqs,
             Uin=Uin_all,
             Uout=Uout_all,
             time=t)
        print(f"Full sweep data saved to {filename}")

        
    def load_data(self, filename="sweep_raw_data.npz"):
        """
        Load full sweep data saved from a previous run.
        Returns: freqs, Uin_all, Uout_all, t
        """
        data = np.load(filename)
        freqs    = data["freqs"]
        Uin_all  = data["Uin"]
        Uout_all = data["Uout"]
        t        = data["time"]

        return freqs, Uin_all, Uout_all, t




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



def measure_sweep_raw(myDAQ, analyzer, freqs, amplitude=1.0, repeats=3, filename="sweep_raw_data.npz"):
    """
    Acquire ALL Uin and Uout waveforms first.
    Returns:
        freqs                      – array of frequencies
        Uin_all[freq_index, rep]  – stored input waveforms
        Uout_all[freq_index, rep] – stored output waveforms
        t                         – time array for each waveform
    """

    N = len(freqs)
    M = repeats
    L = myDAQ.samps_per_chan

    # Allocate arrays
    Uin_all = np.zeros((N, M, L))
    Uout_all = np.zeros((N, M, L))

    print(f"Starting sweep of {N} frequencies with {M} repeats.\n")

    for i, f in enumerate(freqs):
        print(f"Freq {f:.1f} Hz  ({i+1}/{N})")

        for r in range(M):
            sine = myDAQ.gen_sine(frequency=f, amplitude=amplitude)
            response = myDAQ.write_and_read(sine, ao_channel="ao0",
                                            ai_channels_str="ai0, ai1")

            # Channels: response[0] = Uout, response[1] = Uin
            Uout_all[i, r] = response[0]
            Uin_all[i, r]  = response[1]

    t = np.arange(L) / myDAQ.sample_rate
    #SAVE EVERYTHING 
    myDAQ.save_data(filename,
                    freqs=freqs,
                    Uin_all=Uin_all,
                    Uout_all=Uout_all,
                    t=t)

    return freqs, Uin_all, Uout_all, t

"""
def analyze_sweep_fft(analyzer, freqs, Uin_all, Uout_all):
    
    Performs FFT and transfer-function analysis AFTER all data was collected.

    Returns:
        H_mean[f]   – mean |H|
        H_std[f]    – std of |H|
        PH_mean[f]  – mean phase (radians)
    

    N, M, L = Uin_all.shape
    H_vals = np.zeros((N, M))
    PH_vals = np.zeros((N, M))



    # ------------------------------
    # Compute |H| and phase for all points
    # ------------------------------
    for i, f in enumerate(freqs):
        for r in range(M):
            Uin = Uin_all[i, r]
            Uout = Uout_all[i, r]

            fft_in, f_axis = analyzer.compute_fft(Uin)
            fft_out, _     = analyzer.compute_fft(Uout)

            # Find nearest FFT index
            idx = np.argmin(np.abs(f_axis - f))

            H_vals[i, r]  = np.abs(fft_out[idx]) / np.abs(fft_in[idx])
            PH_vals[i, r] = np.angle(fft_out[idx]) - np.angle(fft_in[idx])





    
    # Unwrap phase along repeats axis
    PH_vals = np.unwrap(PH_vals, axis=1)

    H_mean = np.mean(H_vals, axis=1)
    H_std  = np.std(H_vals, axis=1)
    PH_mean = np.mean(PH_vals, axis=1)

    return H_mean, H_std, PH_mean



"""
def analyze_sweep_fft(analyzer, freqs, Uin_all, Uout_all, threshold=100):
    """
    Performs FFT and system identification AFTER loading stored data.
    Automatically removes outliers where |H| deviates more than
    'threshold × mean'.
    """
    N, M, L = Uin_all.shape

    H_vals  = np.zeros((N, M))
    PH_vals = np.zeros((N, M))

    # ------------------------------
    # Compute |H| and phase for all points
    # ------------------------------
    for i, f in enumerate(freqs):
        for r in range(M):

            Uin  = Uin_all[i, r]
            Uout = Uout_all[i, r]

            fft_in,  f_axis = analyzer.compute_fft(Uin)
            fft_out, _      = analyzer.compute_fft(Uout)

            idx = np.argmin(np.abs(f_axis - f))

            H_vals[i, r] = np.abs(fft_out[idx]) / np.abs(fft_in[idx])
            PH_vals[i, r] = np.angle(fft_out[idx]) - np.angle(fft_in[idx])

    # ------------------------------
    # OUTLIER REMOVAL
    # ------------------------------
    H_mean_clean = np.zeros(N)
    H_std_clean  = np.zeros(N)
    PH_mean_clean = np.zeros(N)

    for i in range(N):

        H_row = H_vals[i]
        PH_row = PH_vals[i]

        # Provisional mean to detect spikes
        provisional_mean = np.mean(H_row)

        # Identify allowed values
        good_mask = np.abs(H_row) <= threshold * provisional_mean

        # Ensure minimum 1 valid point
        if np.sum(good_mask) == 0:
            print(f"WARNING: All repeats rejected at frequency {freqs[i]:.1f} Hz.")
            # Fallback: use all data
            good_mask = np.ones_like(H_row, dtype=bool)

        # Remove 360° jumps after filtering
        PH_row_clean = np.unwrap(PH_row[good_mask])
        H_row_clean  = H_row[good_mask]

        # Final cleaned values
        H_mean_clean[i]  = np.mean(H_row_clean)
        H_std_clean[i]   = np.std(H_row_clean)
        PH_mean_clean[i] = np.mean(PH_row_clean)

    return H_mean_clean, H_std_clean, PH_mean_clean


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

def estimate_slope(freqs, H_dB):
    """
    Estimate Bode magnitude slope in the roll-off region.
    Returns: slope_dB_per_decade
    """

    # Convert frequency to log10 scale
    logf = np.log10(freqs)

    # Compute derivative (rough slope estimation)
    dH = np.gradient(H_dB, logf)

    # Identify the region where slope is strongly negative or positive
    # Threshold: |slope| > 5 dB/decade
    slope_region = np.where(np.abs(dH) > 5)[0]

    if len(slope_region) < 5:
        print("WARNING: No clear slope region detected.")
        return None

    # Select that region
    x = logf[slope_region]
    y = H_dB[slope_region]

    # Linear regression
    res = linregress(x, y)

    slope = res.slope      # slope in dB per decade
    intercept = res.intercept

    return slope, intercept, slope_region


def find_magnitude_peak(freqs, H_dB):
    """
    Returns:
        f_peak – frequency where magnitude is maximum
        H_peak – magnitude value
    """

    idx = np.argmax(H_dB)
    return freqs[idx], H_dB[idx], idx


def find_characteristic_freqs(freqs, H_dB):
    """
    Finds cutoff / resonant frequencies based on -3 dB rule.
    Works for low-pass, high-pass, band-pass, band-stop.
    
    Returns:
        dict containing:
            f_c_low
            f_c_high
            f_peak
            filter_type
    """

    # Find maximum magnitude
    f_peak, H_peak, idx_peak = find_magnitude_peak(freqs, H_dB)

    # -3 dB threshold
    cutoff_level = H_peak - 3

    # Find where magnitude drops below cutoff level
    below = H_dB <= cutoff_level
    #checks when the cutoff line is reached, if hit once 1 low or highpass filter, if more time crossed it's a band or gap
    crossings = np.where(np.diff(below.astype(int)) != 0)[0]

    # Interpret based on number of crossings
    if len(crossings) == 0:
        # Possibly low-pass or high-pass where the sweep didn't reach the roll-off
        return {
            "filter_type": "Unknown (no cutoff found)",
            "f_peak": f_peak,
            "f_c_low": None,
            "f_c_high": None
        }

    elif len(crossings) == 1:
        # Single cutoff → low-pass or high-pass
        f_c = freqs[crossings[0]]
        if idx_peak < crossings[0]:
            ftype = "Low-pass"
        else:
            ftype = "High-pass"

        return {
            "filter_type": ftype,
            "f_peak": f_peak,
            "f_c_low": f_c,
            "f_c_high": None
        }

    elif len(crossings) >= 2:
        # Two cutoffs → band-pass or band-stop
        f1 = freqs[crossings[0]]
        f2 = freqs[crossings[-1]]
        # check if peak is between f1 and f2
        if f1 < f_peak < f2:
            ftype = "Band-pass"
        else:
            ftype = "Band-stop"

        return {
            "filter_type": ftype,
            "f_peak": f_peak,
            "f_c_low": f1,
            "f_c_high": f2
        }



if __name__ == "__main__":

    # Initialize MyDAQ and FFT analyzer using same sampling settings
    myDAQ = MyDAQ()
    analyzer = FFTAnalyzer(
        sample_rate=myDAQ.sample_rate, samps_per_chan=myDAQ.samps_per_chan
    )

    mode = input("Run single frequency (s) or full sweep (a)? [s/a]: ")
    
    if mode == "s":
        # --- SINGLE-FREQUENCY MODE ---
        # Quick test to verify correct setup and data flow
        freqs = [100, 500, 1000]
        measure_sweep_raw(
        myDAQ, analyzer, freqs, amplitude=1.0, repeats=3
        )
    
    
    elif mode == "a":

        freqs = np.logspace(1, 4.3, 30)

        # 1. Acquire all data first
        measure_sweep_raw(
            myDAQ, analyzer,
            freqs=freqs,
            amplitude=1.0,
            repeats=3,
            filename="sweep_raw_data.npz")
        
        #read out data from saved file
        freqs, Uin_all, Uout_all, t = myDAQ.load_data("sweep_raw_data.npz")


        # 2. Analyze using FFT afterwards
        H_mean, H_std, PH_mean = analyze_sweep_fft(
        analyzer, freqs, Uin_all, Uout_all
        )

        # 3. Fit and plot
        fit_and_plot(freqs, H_mean, H_std, PH_mean)
        
        
        
        #finding the filter stuff:
        # mag in dB
        H_dB = 20 * np.log10(H_mean)

        # 1. Slope Estimation
        slope, intercept, region = estimate_slope(freqs, H_dB)
        print(f"Slope: {slope:.1f} dB/decade → Order ~ {abs(slope)/20:.1f}")

        # 2. Maximum
        f_peak, H_peak, idx = find_magnitude_peak(freqs, H_dB)
        print(f"Peak magnitude at {f_peak:.1f} Hz: {H_peak:.2f} dB")

        # 3. Characteristic Frequencies
        info = find_characteristic_freqs(freqs, H_dB)

        print("\nDetected filter type:", info["filter_type"])
        print("Peak frequency:", info["f_peak"])
        print("Lower cutoff:", info["f_c_low"])
        print("Upper cutoff:", info["f_c_high"])



        

