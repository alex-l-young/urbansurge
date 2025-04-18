% Fault detectability.

close all
clear

% Load in fault table.
fault_table_fp = "C:\Users\ay434\Documents\urbansurge\analysis\lab_system\Output\diameter_17_ramp_3in_n20_60min.csv";
fault_table = readtable(fault_table_fp);

% Datetime.
t = fault_table.datetime(fault_table.fault_value==0);

% Fault values.
fault_values = unique(fault_table.fault_value);

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
node_id = 59;
node_flood_col_mask = startsWith(fault_table.Properties.VariableNames, strcat("Depth_node_", string(node_id)));

% Node flooding columns.
% node_flood_col_mask = startsWith(fault_table.Properties.VariableNames, 'Flood_node_'); 
% node_flood_cols = fault_table(:,node_flood_col_mask);

% Modeled "healthy" depth.
model_depth = fault_table(fault_table.fault_value==0, node_flood_col_mask);
model_depth = table2array(model_depth);

cumu_node_residuals = zeros(numel(fault_values),numel(peak_idx)-1);
for i = 1:numel(fault_values)
    fault_value = fault_values(i);

    meas_depth = fault_table(fault_table.fault_value==fault_value, node_flood_col_mask);
    meas_depth = table2array(meas_depth);

    % Residuals.
    node_residuals = model_depth - meas_depth;

    % Loop through precip peak indices.
    for j = 1:numel(peak_idx)-1
        % Indices over which to calculate cumulative node flooding.
        peak_start_idx = peak_idx(j);
        peak_end_idx = peak_idx(j+1);

        % Cumulative node residuals between peaks.
        node_peak_res = node_residuals(peak_start_idx:peak_end_idx,:);
%         cumu_node_res = sum(trapz(node_peak_res,1) .* 5 .* 60, 2);
        cumu_node_res = max(node_peak_res);

        % Save to array.
        cumu_node_residuals(i,j) = cumu_node_res;
    end
end

% Subtract out the no-fault row.
% node_flooding = node_flooding - repmat(node_flooding(1,:), size(node_flooding,1), 1);

%% Plot severity.
close all

copper_map = copper(numel(peak_idx)-1);  % Generate copper colormap
color_map = colorGradient([2, 2, 163]./255, [255, 112, 112]./255, numel(peak_idx)-1);

figure()
for i = 1:numel(peak_idx)-1
    plot_fault_values = fault_values ./ 1 .* 100;
    plot(plot_fault_values, cumu_node_residuals(:,i), 'LineWidth', 1,...
        'Color', color_map(i,:))

    % Add label along each line
    if max(cumu_node_residuals(:,i) > 0)
        text(plot_fault_values(end), cumu_node_residuals(end,i),...
            sprintf('P=%0.2f in.', Ppeak(i)), 'Color', color_map(i,:));
    end
    hold on
end
xlim([0 90])
xlabel('Blockage Severity (%)')
ylabel('Cumulative Residual')
set(gca,'FontSize',15)

% Change in flood magnitude for a change in fault magnitude.
node_flooding_diff = diff(cumu_node_residuals,1,1);
dFlood_dFault = node_flooding_diff ./ repmat(diff(plot_fault_values), 1, size(node_flooding_diff,2));
dFlood_dFault = dFlood_dFault(cumu_node_residuals(:,1:end-1) > 0);
mean(dFlood_dFault)
figure()
% for i = 1:numel(peak_idx)-1
%     hold on
%     scatter(plot_fault_values(1:end-1), node_flooding_diff(:,i), 'b')
% end
histogram(dFlood_dFault, 21)
% xlabel('$\theta$', 'Interpreter','latex')
xlabel('$\Delta \epsilon / \Delta \theta$', 'Interpreter','latex')
set(gca, 'FontSize', 16)

% Return period shift due to severity.
return_period = logspace(-2,1,numel(Ppeak));
figure()
for i = 1:numel(fault_values)-1
    plot_fault_values = fault_values ./ 1 .* 100;
    plot(return_period, cumu_node_residuals(i,:), 'LineWidth', 1,...
        'Color', color_map(i,:))

    % Add label along each line
    if max(cumu_node_residuals(:,i) > 0)
        text(return_period(end), cumu_node_residuals(i,end),...
            sprintf('Fault=%0.0f', plot_fault_values(i)), 'Color', color_map(i,:));
    end
    hold on
end
% xlim([0 90])
xlabel('Return Period (yr)')
ylabel('Node Flooding (cf)')
set(gca,'FontSize',15)

% Node flooding contour plot.
node_flooding_contour = cumu_node_residuals;
node_flooding_contour(node_flooding_contour <= 0) = -500;

figure()
[C,h] = contourf(fault_values .* 100, Ppeak, node_flooding_contour');%, 0:500:5000);
colormap(flipud(hot))
clabel(C,h)
c = colorbar;
c.Label.String = 'Flooding';
c.Label.FontSize = 15;
% caxis([0 5000])
xlabel('Blockage %')
ylabel('Precip. Intensity')
set(gca, 'FontSize', 15)

figure()
[C,h] = contourf(fault_values .* 100, return_period, node_flooding_contour');%, 0:500:5000);
colormap(flipud(hot))
clabel(C,h)
xlabel('Blockage %')
ylabel('Return Period')
set(gca, 'FontSize', 15)
