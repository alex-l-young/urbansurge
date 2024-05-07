% Flood sensitivity.

close all
clear

% Load in fault table.
fault_table_fp = "C:\Users\ay434\Documents\urbansurge\analysis\lab_system\Output\diameter_19_ramp_3in_n20_60min.csv";
fault_table = readtable(fault_table_fp);

% Datetime.
t = fault_table.datetime(fault_table.fault_value==0);

% Fault values.
fault_values = unique(fault_table.fault_value);
prc_fault_values = fault_values ./ 1 .* 100; % Percent fault values.

% Precipitation.
P = fault_table.prcp(fault_table.fault_value==0);

% Find peak precip vales.
Ppeak = findpeaks(P);

% Peak indices.
peak_log = ismember(P, Ppeak);
peak_idx_diff = diff(peak_log);
peak_idx = find(peak_idx_diff==1);

% Add on final index of array.
peak_idx = [peak_idx; length(P)];

% Total node flooding for each peak.
node_id = 38;
% node_flood_col_mask = startsWith(fault_table.Properties.VariableNames, strcat("Flood_node_", string(node_id)));

% Node flooding columns.
node_flood_col_mask = startsWith(fault_table.Properties.VariableNames, 'Flood_node_'); 
node_flood_cols = fault_table.Properties.VariableNames(node_flood_col_mask);
node_names = cellfun(@(str) strrep(str, 'Flood_node_', ''), node_flood_cols, 'UniformOutput', false);
node_names_num = cellfun(@str2num, node_names);

% Loop through columns and compute node flood sensitivity.
node_flood_sensitivity = zeros(1,numel(node_flood_cols));
for n = 1:numel(node_flood_cols)
    % Node flood column to process.
    node_flood_col = node_flood_cols(n);
    node_flood_col

    % Empty array to save node flooding values.
    node_flooding = zeros(numel(fault_values),numel(peak_idx)-1);
    for i = 1:numel(fault_values)
        fault_value = fault_values(i);
    
        % Select rows that correspond to the fault value.
        node_flood = fault_table(fault_table.fault_value==fault_value, ...
                node_flood_col);
        node_flood = table2array(node_flood);
    
        % Loop through precip peak indices.
        for j = 1:numel(peak_idx)-1
            % Indices over which to calculate cumulative node flooding.
            peak_start_idx = peak_idx(j);
            peak_end_idx = peak_idx(j+1);
    
            % Cumulative node flooding between peaks.
            node_peak_flood = node_flood(peak_start_idx:peak_end_idx,:);
            cumu_node_flood = sum(trapz(node_peak_flood,1) .* 5 .* 60, 2);
    
            % Save to array.
            node_flooding(i,j) = cumu_node_flood;
        end
    end

    % Compute dFlood.
    node_flooding_diff = diff(node_flooding,1,1);

    % Change in flood with change in fault severity.
    dFlood_dFault = node_flooding_diff ./ repmat(diff(prc_fault_values),...
        1, size(node_flooding_diff,2));
    dFlood_dFault = dFlood_dFault(node_flooding(:,1:end-1) > 0);

    % Overall sensitivity is the mean sensitivity across all precip rates.
    if numel(dFlood_dFault) > 0
        node_flood_sensitivity(n) = mean(dFlood_dFault);
    else
        node_flood_sensitivity(n) = 0;
    end
end

%% Plotting.
figure()
bar(node_names_num, node_flood_sensitivity)