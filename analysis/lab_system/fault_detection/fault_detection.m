% Fault detection.

close all
clear

% Load in fault database.
fault_table_fp = "C:\Users\ay434\Documents\urbansurge\analysis\lab_system\Output\diameter_19_ramp_3in_n20_60min.csv";
fault_table = readtable(fault_table_fp);

% Fault values.
fault_values = unique(fault_table.fault_value);
prc_fault_values = fault_values ./ 1 .* 100; % Percent fault values.

% Datetime.
t = fault_table.datetime(fault_table.fault_value==0);

% Adjacency matrix for epaswmm network.
Atable = readtable("C:\Users\ay434\Documents\urbansurge\analysis\lab_system\adjacency_matrix.csv");

% Strip "node_" off the node name.
node_names = Atable.Properties.VariableNames;
G_node_names = cellfun(@(str) strrep(str, 'node_', ''), node_names, 'UniformOutput', false);
G_node_names_num = cellfun(@str2num, G_node_names);

% Create matrix from Atable.
A = table2array(Atable);

% Grab edge names from Atable.
edge_names = [];
for i = 1:size(A,1)
    for j = 1:size(A,2)
        if A(i,j) > 0
            edge_names = [edge_names A(i,j)];
        end
    end
end

% Set all values in A greater than 0 to 1.
A(A > 0) = 1;

% Directed graph of the network.
G = digraph(A);
G.Nodes.Name = G_node_names';

figure()
p = plot(G, 'LineWidth', 3);
colormap('autumn');

%% Node depth residuals.
sensor_node = 8;

% Node depth column.
node_depth_str = strcat("Depth_node_", string(sensor_node));

% Modeled "healthy" depth.
model_depth = fault_table(fault_table.fault_value==0, node_depth_str);
model_depth = table2array(model_depth);

% Measured "faulty" depth.
for i = 1:numel(fault_values)
    meas_depth = fault_table(fault_table.fault_value==fault_values(i), node_depth_str);
    meas_depth = table2array(meas_depth);

    % Residuals.
    node_residuals = model_depth - meas_depth;

end


% Plot the residuals.
close all
figure()
subplot(211)
hold on
plot(t, meas_depth, 'DisplayName', 'Meas.')
plot(t, model_depth, 'DisplayName', 'Model')
hold off
legend()
set(gca, 'FontSize', 15)

subplot(212)
plot(t, node_residuals)
set(gca, 'FontSize', 18)

%% Link velocity residuals.
sensor_link = 39;
fault_value = 0.2;

% Link velocity column.
link_velocity_str = strcat("Velocity_link_", string(sensor_link));

% Modeled "healthy" velocity.
model_link_velocity = fault_table(fault_table.fault_value==0,...
    link_velocity_str);
model_link_velocity = table2array(model_link_velocity);

% Measured "faulty" velocity.
meas_link_velocity = fault_table(fault_table.fault_value==fault_value,...
    link_velocity_str);
meas_link_velocity = table2array(meas_link_velocity);

% Link depth column.
link_depth_str = strcat("Depth_link_", string(sensor_link));

% Modeled "healthy" depth.
model_link_depth = fault_table(fault_table.fault_value==0,...
    link_depth_str);
model_link_depth = table2array(model_link_depth);

% Measured "faulty" depth.
meas_link_depth = fault_table(fault_table.fault_value==fault_value,...
    link_depth_str);
meas_link_depth = table2array(meas_link_depth);

% Link residuals.
link_velocity_residuals = model_link_velocity - meas_link_velocity;
link_depth_residuals = model_link_depth - meas_link_depth;

% Plot the residuals.
close all
figure()
subplot(211)
hold on
plot(t, meas_link_velocity, 'DisplayName', 'Meas.')
plot(t, model_link_velocity, 'DisplayName', 'Model')
hold off
legend()
set(gca, 'FontSize', 15)

subplot(212)
plot(t, link_velocity_residuals)
set(gca, 'FontSize', 18)

figure()
subplot(211)
hold on
plot(t, meas_link_depth, 'DisplayName', 'Meas.')
plot(t, model_link_depth, 'DisplayName', 'Model')
hold off
legend()
set(gca, 'FontSize', 15)

subplot(212)
plot(t, link_depth_residuals)
set(gca, 'FontSize', 18)
