%{
data_collection.m

Collect height sensor data for a trial
Write data to spreadsheet
%}

close all
clear

i = 0; % trial number (modify to write to new spreadsheet)
fs = 4000; % daq sampling rate (Hz)
dt = 5; % trial length (s)

dq = daq("ni");
dq.Rate = fs;

% add inputs
ch0 = addinput(dq, "Dev1", "ai0", "Voltage"); % bucket sensor
ch0.TerminalConfig = "SingleEnded";

ch1 = addinput(dq, "Dev1", "ai1", "Voltage"); % pipe sensor [###]
ch1.TerminalConfig = "SingleEnded";

% collect data
[data, time, start] = read(dq, seconds(dt), OutputFormat="Matrix");
V_ai0 = data(:,1);
V_ai1 = data(:,2);

% write to spreadsheet
tab = table(time, V_ai0, V_ai1);
date = string(datetime('now'));
path = "sensor_data\";
filename = date + "_" + "sensor_data" + ".csv";
writetable(tab, path+filename, 'WriteMode', 'append');

plot(time, V_ai1)
xlabel('Time (s)')
ylabel('Voltage (V)')