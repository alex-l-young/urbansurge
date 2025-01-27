close all
clear

dq = daq("ni");

dq.Rate = 1000;
ch = addinput(dq, "Dev1", "ai1", "Voltage");
ch.TerminalConfig = "SingleEnded";
f = figure();
data = read(dq, seconds(5));
clf(f);
plot(data.Time, data.Variables)
xlabel('Time (s)')
ylabel('Voltage (V)')

dataM = mean(data.Variables);
stDev = std(data.Variables);
z = abs((data.Variables - dataM))./stDev;

newData = data.Variables;
filtered = newData(z < 4);

figure();
plot(filtered)
xlabel('Time (s)')
ylabel('Voltage (V)')
newMean = mean(filtered)
