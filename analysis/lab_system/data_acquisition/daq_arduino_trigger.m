% DAQ output trigger to arduino.

close all
clear

i = 0; % trial number (modify to write to new spreadsheet)
dt_sensor = 0.03; % Sensor sampling rate.
fs = 1/dt_sensor; % daq sampling rate (Hz)
dt = 1; % trial length (s)

dq = daq("ni");
dq.Rate = fs;

% Output signal.
ch0_out = addoutput(dq, "Dev1", "ao0", "Voltage");
write(dq, 0);
% write(dq, 3);
% write(dq, 0);

% addoutput(dq, "Dev1", "ao0", "Voltage");
% addclock(dq,"Dev1","ao0",dq.Rate);
% start(dq,"RepeatOutput")
% % ⋮
% write(dq, 1);
% % Device output now repeated while MATLAB continues.
% pause(20)
% stop(dq)