# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:19:16 2025

@author: s4041518
"""
# Installs some basic functions

#%matplotlib inline
###CODE SESSION 1
import nidaqmx as dx
import matplotlib.pyplot as plt
import numpy as np
import time



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
    def save_data(self, filename="session4_last.npy"):
        np.savez(filename, time = self._last_time, voltage = self._last_data)
        print(f"Data saved to {filename}")
        
    def load_data(self, filename= "session2_voices.npy"):
        """
        Load previously saved data file. 
        """
        data = np.load(filename)
        self._last_time = data["time"]
        self._last_data = data["voltage"]
        return self._last_time, self._last_data
        pass



if __name__ == "__main__":
    myDAQ = MyDAQ()
    #myDAQ.set_sample_rate(rate = 10000, samps_per_chan=10000)

    #write sine
    sine = myDAQ.gen_sine(frequency=5, amplitude=2.0)
    myDAQ.write(sine)

    #read back from AI0
    voltages = myDAQ.read(channel ="ai0")
    times = myDAQ.getTimeData()


    #write and read
    response = myDAQ.write_and_read(sine)
    plt.figure(figsize=(20, 10))
    plt.plot(times, sine, label= "output signal")
    plt.plot(times, response, label = "input")
    plt.xlabel("Time (s)", fontsize = 30)
    plt.ylabel("Voltage (V)", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.legend(fontsize = 30)
    plt.show()
