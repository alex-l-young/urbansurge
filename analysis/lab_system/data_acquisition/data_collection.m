%{
data_collection.m

Collect height sensor data for a trial
Write data to spreadsheet
%}

close all
clear

% experimental parameters
fault_level = 0; % 0 for none, 1 for 20%, 2 for 40%
drained = 0; % 0 for not drained, 1 for drained
impulse_length = 1; % 0 is shortest  !!! check how many seconds !!!

% Trigger valve.
%************************************
dq = daq("ni");

% Output trigger signal.
ch0_out = addoutput(dq, "Dev1", "ao0", "Voltage");

% Trigger signal.
pause(1)
write(dq, 0);
write(dq, 4);
pause(0.1)
write(dq, 0);
pause(1)
disp('TRIGGERED')

% Data Collection.
%***********************************
% Clear daq instance and prepare for data collection
clear dq
dq = daq("ni");

% add inputs
ch0 = addinput(dq, "Dev1", "ai0", "Voltage"); % pipe sensor [NAME]
ch0.TerminalConfig = "SingleEnded";

ch1 = addinput(dq, "Dev1", "ai1", "Voltage"); % pipe sensor [NAME]
ch1.TerminalConfig = "SingleEnded";

ch2 = addinput(dq, "Dev1", "ai2", "Voltage"); % bucket sensor
ch2.TerminalConfig = "SingleEnded";

% collect data
dt_sensor = 0.03; % Sensor sampling rate.
fs = 1/dt_sensor; % daq sampling rate (Hz)
dt = 600; % trial length (s)
dq.Rate = fs;
[data, time, start] = read(dq, seconds(dt), OutputFormat="Matrix");
V_ai0 = data(:,1);
V_ai1 = data(:,2);
V_ai2 = data(:,3);

figure()
subplot(311)
plot(time, V_ai0)
xlabel('Time (s)')
ylabel('Voltage (V)')

subplot(312)
plot(time, V_ai1)
xlabel('Time (s)')
ylabel('Voltage (V)')

subplot(313)
plot(time, V_ai2)
xlabel('Time (s)')
ylabel('Voltage (V)')

%% write to spreadsheet
tab = table(time, V_ai0, V_ai1, V_ai2);
date = string(datetime('now', 'Format', 'yyyy-MM-dd''_''HH-mm-ss'));
path = "sensor_data\";
filename = date + "_" + "sensor_data" + ".csv";
writetable(tab, fullfile(path,filename));

%% add to file organization spreadsheet
to_open = "data_organization_updated.csv";
path_open = "data_acquisition\";
tab = table(filename, fault_level, drained, impulse_length);
writetable(tab, to_open, 'WriteMode', 'append');