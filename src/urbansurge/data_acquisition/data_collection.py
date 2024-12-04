import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import os

# Initialize variables
i = 0  # trial number (modify to write to new spreadsheet)
fs = 400  # DAQ sampling rate (Hz)
dt = 1 / fs  # Time step length.
T = 20 # Length of trial.
L = int(T / dt) # Number of samples.

with nidaqmx.Task() as task:
	task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

	task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=L)

	data = task.read(READ_ALL_AVAILABLE, timeout=T + 0.1)

# Time.
t = np.arange(0, T, dt)

plt.plot(t, data)

plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('Sensor Reading')

plt.show()


# # Set up DAQ device (assumed DAQmx compatible device)
# with nidaqmx.Task() as task:
#     task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  # bucket sensor
#     task.ai_channels.add_ai_voltage_chan("Dev1/ai1")  # pipe sensor

#     # Configure the sample rate and timing
#     task.timing.cfg_samp_clk_timing(rate=fs, samps_per_chan=fs * dt)

#     # Collect data (this will collect for the duration of dt seconds)
#     data = task.read(number_of_samples_per_channel=fs * dt)

# # Convert data to numpy array (time will be calculated separately)
# time_array = np.linspace(0, dt, fs * dt)
# V_ai0 = np.array(data[0])
# V_ai1 = np.array(data[1])

# # Prepare data for saving
# date_str = datetime.today().strftime('%Y-%m-%d')
# path = r"C:\Users\rabbi\OneDrive\Documents\Stormwater Research\Data acquisition\sensor_daq_code\sensor_data"
# filename = f"{date_str}_sensor_data_{i}.csv"
# filepath = os.path.join(path, filename)

# # Create a DataFrame for easy handling and saving to CSV
# df = pd.DataFrame({'Time (s)': time_array, 'V_ai0': V_ai0, 'V_ai1': V_ai1})

# # Write the data to CSV (append mode if the file exists)
# # df.to_csv(filepath, mode='a', index=False)

# # Plot the data (example: V_ai1)
# plt.plot(time_array, V_ai1)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.title('Pipe Sensor Data')
# plt.show()
