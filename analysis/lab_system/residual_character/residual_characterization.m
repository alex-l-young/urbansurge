% Characterize no-fault physical system residuals.

close all
clear

% Load in physical system sensor readings.
physical_table_fp = "C:\Users\ay434\Documents\urbansurge\analysis\lab_system\Output\physical_ramp_3in_n20_60min.csv";
physical_table = readtable(physical_table_fp);

% Load in ensemble system sensor readings.
model_table_fp = "C:\Users\ay434\Documents\urbansurge\analysis\lab_system\Output\ensemble_ramp_3in_n20_60min.csv";
model_table = readtable(model_table_fp);

% Number of ensembles.
Nens = size(model_table,1) / size(physical_table,1);

% Datetime.
t = physical_table.datetime;

% Precipitation.
P = physical_table.prcp;

% Find peak precip vales.
Ppeak = findpeaks(P);

% Peak indices.
peak_log = ismember(P, Ppeak);
peak_idx_diff = diff(peak_log);
peak_idx = find(peak_idx_diff==1);

% Add on final index of array.
peak_idx = [peak_idx; length(P)];

%% Plot flow through a link.
close all

link_id = 33;
column_name = strcat('Flow_link_', string(link_id));

% Flow in link.
phy_link_flow = table2array(physical_table(:,column_name));
mod_link_flow = table2array(model_table(:,column_name));

% Compute residuals.
residuals = mod_link_flow - phy_link_flow;

figure()
plot(t, residuals)