%{
data_collection.m

Collect height sensor data for a trial
Write data to spreadsheet
%}

close all
clear

i = 0; % trial number (modify to write to new spreadsheet)
dt_sensor = 0.03; % Sensor sampling rate.
fs = 1/dt_sensor; % daq sampling rate (Hz)
dt = 40; % trial length (s)

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

figure()
subplot(211)
plot(time, V_ai0)
xlabel('Time (s)')
ylabel('Voltage (V)')

subplot(212)
plot(time, V_ai1)
xlabel('Time (s)')
ylabel('Voltage (V)')

%% write to spreadsheet
tab = table(time, V_ai0, V_ai1);
date = string(datetime('now', 'Format', 'yyyy-MM-dd''_''HH-mm-ss'));
path = "sensor_data\";
filename = date + "_" + "sensor_data" + ".csv";
writetable(tab, fullfile(path,filename));