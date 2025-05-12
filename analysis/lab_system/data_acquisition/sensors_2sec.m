close all
clear

dq = daq("ni");

% Sample duration.
duration = 3; % Seconds

% DAQ channel.
channel = 'ai2';

% DAQ sampling rate.
dt_sensor = 0.03; % Sensor sampling rate.
fs = 1/dt_sensor; % daq sampling rate (Hz)
dt = 60; % trial length (s)
dq.Rate = fs;

ch = addinput(dq, "Dev1", channel, "Voltage");
ch.TerminalConfig = "SingleEnded";
f = figure();
data = read(dq, seconds(duration));
clf(f);
plot(data.Time, data.Variables)
xlabel('Time (s)')
ylabel('Voltage (V)')

dataM = mean(data.Variables)
stDev = std(data.Variables);
z = abs((data.Variables - dataM))./stDev;

%newData = data.Variables;
%filtered = newData(z < 4);

%figure();
%plot(filtered)
%xlabel('Time (s)')
%ylabel('Voltage (V)')
%newRange = max(filtered) - min(filtered)
%newMean = mean(filtered)

%figure()
%plot()