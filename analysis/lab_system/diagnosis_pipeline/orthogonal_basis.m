% Unit vector mean.

close all
clear

% Generate n random vectors.
n = 10;
X = [unifrnd(1, 2, 1, n); unifrnd(1.5, 2, 1, n)];
X2 = [unifrnd(1, 2, 1, n); unifrnd(0, 0.5, 1, n)];

% Unit vectors.
U = X ./ vecnorm(X);
U2 = X2 ./ vecnorm(X2);

% Mean of unit vectors.
Umean = mean(U, 2) ./ vecnorm(mean(U, 2));
Umean2 = mean(U2, 2) ./ vecnorm(mean(U2, 2));

% Observe some new data.
Xobs = unifrnd(1, 2, 2, 1);
Xobs = [unifrnd(1, 2, 1, 1); unifrnd(0.4, 0.7, 1, 1)];
Uobs = Xobs ./ vecnorm(Xobs);

% Project onto mean unit vectors.
pr = dot(Uobs, Umean) / dot(Umean, Umean)
pr2 = dot(Uobs, Umean2) / dot(Umean2, Umean2)

figure()
subplot(121)
hold on
for i = 1:n
    plot([0, X(1,i)], [0, X(2,i)], 'color', [134, 167, 207] ./ 255)
    plot([0, U(1,i)], [0, U(2,i)], 'b')

    plot([0, X2(1,i)], [0, X2(2,i)], 'color', [207, 134, 134] ./ 255)
    plot([0, U2(1,i)], [0, U2(2,i)], 'r')
end
plot([0, Umean(1)], [0, Umean(2)], 'k', 'LineWidth', 2)
plot([0, Umean2(1)], [0, Umean2(2)], 'k', 'LineWidth', 2)
plot([0, Uobs(1)], [0, Uobs(2)], 'color', [3, 143, 3] ./ 255, 'LineWidth', 2)
xlabel('$t_1$', 'interpreter', 'latex')
ylabel('$t_2$', 'interpreter', 'latex')
set(gca, 'fontsize', 15)

subplot(122)
% Define the axis limits
x = linspace(0, 10, 100); % Define the x-axis range
y = x; % Define the 1:1 line
hold on
% Shade the area above the 1:1 line
fill([x, fliplr(x)], [y, 10 * ones(size(y))], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% Shade the area below the 1:1 line
fill([x, fliplr(x)], [y, zeros(size(y))], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
scatter(pr, pr2, 50, [3, 143, 3] ./ 255, 'filled')
plot([0 1], [0 1], 'k--')
hold off
xlim([0 1])
ylim([0 1])
xlabel('$\frac{\vec x_{obs} \cdot \vec{\overline{u}}_1}{\vec{\overline{u}}_1 \cdot \vec{\overline{u}}_1}$', 'interpreter', 'latex')
ylabel('$\frac{\vec x_{obs} \cdot \vec{\overline{u}}_2}{\vec{\overline{u}}_2 \cdot \vec{\overline{u}}_2}$', 'interpreter', 'latex')
set(gca, 'fontsize', 16)