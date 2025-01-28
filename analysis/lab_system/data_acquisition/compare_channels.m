% Compare sensor readings across DAQ channels.

close all
clear

dq = daq("ni");

% Sample duration.
T = 5; % Seconds

% DAQ sampling rate.
dt_sensor = 0.03; % Sensor sampling rate.
fs = 1/dt_sensor; % daq sampling rate (Hz)
dq.Rate = fs;

% Add all available channels.
ch1 = addinput(dq, "Dev1", "ai0", "Voltage");
ch1.TerminalConfig = "SingleEnded";
ch2 = addinput(dq, "Dev1", "ai1", "Voltage");
ch2.TerminalConfig = "SingleEnded";
ch3 = addinput(dq, "Dev1", "ai2", "Voltage");
ch3.TerminalConfig = "SingleEnded";

% Read data.
[data, time, start] = read(dq, seconds(T), OutputFormat="Matrix");
V_ai0 = data(:,1);
V_ai1 = data(:,2);
V_ai2 = data(:,3);

figure()
hold on
% plot(time, (V_ai0 - mean(V_ai0)) ./ std(V_ai0), 'LineWidth', 1, 'DisplayName', 'ai0')
% plot(time, (V_ai1 - mean(V_ai1)) ./ std(V_ai1), 'LineWidth', 1, 'DisplayName', 'ai1')
% plot(time, (V_ai2 - mean(V_ai2)) ./ std(V_ai2), 'LineWidth', 1, 'DisplayName', 'ai2')
plot(time, (V_ai0 - mean(V_ai0)), 'LineWidth', 1, 'DisplayName', 'ai0')
plot(time, (V_ai1 - mean(V_ai1)), 'LineWidth', 1, 'DisplayName', 'ai1')
plot(time, (V_ai2 - mean(V_ai2)), 'LineWidth', 1, 'DisplayName', 'ai2')
hold off
xlabel('Time (s)')
ylabel('Voltage (V)')
legend()
set(gca, 'FontSize', 15)


% dataM = mean(data.Variables);
% stDev = std(data.Variables);
% z = abs((data.Variables - dataM))./stDev;
% 
% newData = data.Variables;
% filtered = newData(z < 4);
% 
% figure();
% plot(filtered)
% xlabel('Time (s)')
% ylabel('Voltage (V)')
% newRange = max(filtered) - min(filtered)
% newMean = mean(filtered)
% 
% figure()
% plot()
