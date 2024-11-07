close all
clear

dq = daq("ni");

dq.Rate = 100;
ch = addinput(dq, "Dev1", "ai0", "Voltage");
ch.TerminalConfig = "SingleEnded";
k=1;
L = 1000;
f = figure();
data_Time = [0];
data_Variables = [0];
while true
data = read(dq, seconds(0.1));
data_Time = [data_Time; data.Time+data_Time(end)];
data_Variables = [data_Variables; data.Variables];
if numel(data_Time) > L
    data_Time = data_Time(end - L:end);
    data_Variables = data_Variables(end - L:end);
end
clf(f);
plot(data_Time, data_Variables)
%xlabel('Time (s)')
%ylabel('Voltage (V)')

if k == 10e3
    break
end
k = k + 1;
pause(0.01)
end

%s.DurationInSeconds = 10;
%s.NotifyWhenDataAvailableExceeds = 100;
dq.ScansRequiredFcn = @writeMoreData;
start(dq,"Continuous");
disp('Go')
data = read(dq, seconds(2));
dq.wait();
%delete(lh)

%);
%ylabel("Voltage (V)")