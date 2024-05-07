% Generate precipitation time series.

close all
clear

% Time.
T = 24 * 60; % Minutes.
dt = 5; % Minutes.
Nt = T / dt;
t = 0:dt:T;

%% Increasing impulses.
impulse_max = 1.2; % Maximum impulse value.
Nimpulse = 20; % Number of impulses.
impulse_spacing = 60; % Spacing between impulses [minutes].

% Impulse values.
impulse_values = linspace(0, impulse_max, Nimpulse);

% Precipitation time series.
P = zeros(size(t));
P(1:impulse_spacing/dt:Nimpulse*impulse_spacing/dt) = impulse_values;

figure()
plot(t, P)

%% Random impulses.
impulse_max = 1;
Nimpulse = 20;
impulse_spacing = 60;

% Impulse values.
impulse_values = unifrnd(0, impulse_max, 1, Nimpulse);

% Precipitation time series.
P = zeros(size(t));
P(1:impulse_spacing/dt:Nimpulse*impulse_spacing/dt) = impulse_values;

figure()
plot(t, P)

%% Save precipitation.
time_start = datetime('2020-01-01 00:00');
time_end = time_start + minutes(T);
datetimes = time_start:minutes(dt):time_end;
times = timeofday(datetimes)';
dates = datetimes';
dates.Format = 'MM/dd/yyyy';

% Precipitation table.
P_table = table(dates, times, P');
writetable(P_table, "C:\Users\ay434\Documents\urbansurge\analysis\lab_system\precipitation\impulse_unif_1in_n20_60min.csv")