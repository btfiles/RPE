%% Visualizing MLE2 sim results
%
% This file ought to reproduce the figures in the submitted paper for HCII
% 2018 by Canady et al.
%

%% Settings

mthd_lbl = {'win', 'reg', 'mle'};
col_lbl = {'HR', 'FAR', 'Mu', 'Sigma', 'Tau'};

clrs = [82, 43, 114; 78, 146, 49; 170, 127, 57]./255; % nice color triad

% keep the data on a reasonable scale by truncating extrema, 1 row per parameter
lims = [-0.1 0.15; 
    -0.02 0.01; 
    -0.2 0.2; 
    -0.2 0.2; 
    -0.2 0.2]; 

BIGFIG = [2 28 1278 1487]; % For the larger-figured figure.

jitter = 0.8; % jitter in boxplot outliers

%% Loading & combining data
%
% Simulations were run in three chunks. One of 300 reps, one local of 100
% reps and one remote (at APG/BIERS lab) of 100 reps. 

data_folder = '/home/bfiles/matlab/rpe_pkg/MLE Sim 2 Final/';
file_names = {'Final300.mat', 'Final100_local.mat', 'Final100_remote.mat'};

% Some variables have 1 entry per simulation. We'll need to stack those
% variables we need to stack:
stack_varnames = {'map_estimates', 'reg_estimates', 'win_estimates'};

% Some variables should be common accross runs. We need one of those.
% variables we need one of:
nonce_varnames = {'DIFF_THRESHOLD', 'conditions'};

file_data = cell(size(file_names));
for iFile = 1:numel(file_names)
    file_data{iFile} = load(fullfile(data_folder, file_names{iFile}), ...
        stack_varnames{:}, nonce_varnames{:});
end

d = struct();
for iNonce = 1:numel(nonce_varnames)
    % check they all agree:
    nvn = nonce_varnames{iNonce};
    if ~all(cellfun(@(c) all(reshape(c.(nvn)==file_data{1}.(nvn), [], 1)), file_data))
        error('Disagreement on nonce variables!');
    end
    
    % put it in the master
    d.(nvn) = file_data{1}.(nvn);
end

for iStack = 1:numel(stack_varnames)
    % Stack the stackers
    svn = stack_varnames{iStack};
    tmp = cell(size(file_data));
    for jFile = 1:numel(file_data)
        tmp{jFile} = file_data{jFile}.(svn);
    end
    d.(svn) = cat(ndims(tmp{1}), tmp{:});
end

%% boxplot errors
% This produces a large figure with one row per parameter (hr, far, mu,
% sig, tau) and visualizes the error for each method & simulated condition.

map_errs = d.map_estimates-d.conditions;
reg_errs = d.reg_estimates-d.conditions;
win_errs = d.win_estimates-d.conditions;

fbig = figure('Name', 'AllBoxplots', 'Position', BIGFIG);
ncond = size(d.conditions, 1);
% Inelegant, but the window method does not estimate the RT distribution:
axl = zeros(1,5);
for i = 1:2
    p_map_errs = squeeze(map_errs(:, i, :))';
    p_reg_errs = squeeze(reg_errs(:, i, :))';
    p_win_errs = squeeze(win_errs(:, i, :))';
    
    X = zeros(size(p_map_errs, 1), size(p_map_errs, 2) + ...
        size(p_reg_errs, 2) + size(p_win_errs, 2));
    
    X(:, 1:3:end) = p_win_errs;
    X(:, 2:3:end) = p_reg_errs;
    X(:, 3:3:end) = p_map_errs;
    
    Gmethod = repmat([1 2 3],1, size(X, 2)/3);
    
    Gcond = reshape(repmat(1:ncond, 3, 1), 1, []);
    subplot(5,1,i);
    axl(i) = gca();
    hold on;
    pos = 1:(4*ncond);
    pos(4:4:end) = [];
    bp = boxplot(X, {Gcond, Gmethod}, 'plotstyle', 'traditional', ...
        'medianstyle', 'line', 'colors', clrs, 'datalim', lims(i,:), ...
        'notch', 'on', 'positions', pos, 'symbol', '.');
    plot(xlim(), [0 0], 'k-');
    
    set(bp(1:2, :), 'LineStyle', '-'); % dashed lines weren't working
    
    % Work out ticks and labels
    tickx = 2:4:(4*ncond);
    set(gca, 'XTick', tickx, 'XTickLabel', ' ');

    legend(bp(5, 1:3), mthd_lbl, 'location', 'best');
    ylabel([col_lbl{i} ' error']);
end

% Now the RT distribution
for i = 3:5
    p_map_errs = squeeze(map_errs(:, i, :))';
    p_reg_errs = squeeze(reg_errs(:, i, :))';
    
    X = zeros(size(p_map_errs, 1), size(p_map_errs, 2) + size(p_reg_errs, 2));
    
    X(:, 1:2:end) = p_reg_errs;
    X(:, 2:2:end) = p_map_errs;
    
    Gmethod = repmat([1 2],1, size(X, 2)/2);
    
    Gcond = reshape(repmat(1:25, 2, 1), 1, []);
    subplot(5,1,i);
    axl(i) = gca();
    hold on;
    
    pos = reshape(bsxfun(@plus, [1.5; 2.5], 0:4:96), 1, []);
    bp = boxplot(X, {Gcond, Gmethod}, 'plotstyle', 'traditional', 'symbol', '.', ...
        'medianstyle', 'line', 'colors', clrs(2:end, :), 'datalim', lims(i,:), ...
        'notch', 'on',  'positions', pos);
    plot(xlim(), [0 0], 'k-');
    
    set(bp(1:2, :), 'LineStyle', '-');
    
    % Work out ticks and labels
    tickx = 2:4:100;
    set(gca, 'XTick', tickx, 'XTickLabel', ' ');
    
    legend(bp(5, 1:2), mthd_lbl(2:end));
    
    ylabel([col_lbl{i} ' error']);
end
xtl = sprintf('(%2.1f, %2.1f)\n', d.conditions(:,1:2)'.*100);
set(gca, 'XTick', tickx, 'XTickLabel', xtl, 'XTickLabelRotation', 45);
linkaxes(axl, 'x');
xlabel('Simulated Condition (HR, FAR)');

%% Overall summary
fsmall = figure('name', 'OverallBoxplots');
for i = 1:2
    subplot(2,2,i);
    p_map_errs = squeeze(map_errs(:, i, :))';
    p_reg_errs = squeeze(reg_errs(:, i, :))';
    p_win_errs = squeeze(win_errs(:, i, :))';
    
    bp = boxplot([p_win_errs(:) p_reg_errs(:) p_map_errs(:)], 'colors', clrs, ...
        'symbol', '.', 'jitter', jitter, 'datalim', lims(i,:), ...
        'labels', mthd_lbl, 'notch', 'on');
    set(bp(1:2, :), 'LineStyle', '-');
    hold on;
    m = mean([p_win_errs(:) p_reg_errs(:) p_map_errs(:)]);
    for j = 1:3
        plot(j, m(j), 'd', 'Color', clrs(j, :), ...
            'MarkerFaceColor', clrs(j,:));
    end
    plot(xlim(), [0 0], 'k-');
    title(col_lbl{i})
    ylabel('error (proportion)');
end

for i = 3:5
    p_map_errs = squeeze(map_errs(:, i, :))';
    p_reg_errs = squeeze(reg_errs(:, i, :))';
    
    subplot(2,3,i+1);
    bp = boxplot([p_reg_errs(:) p_map_errs(:)], 'colors', clrs(2:end, :), ...
        'symbol', '.', 'jitter', jitter, 'datalim', lims(i,:), ...
        'labels', mthd_lbl(2:end), 'notch', 'on');
    set(bp(1:2, :), 'LineStyle', '-');
    hold on;
    
    m = mean([p_reg_errs(:) p_map_errs(:)]);
    for j = 1:2
        plot(j, m(j), 'd', 'Color', clrs(j+1, :), 'MarkerFaceColor', clrs(j+1,:));
    end
    plot(xlim(), [0 0], 'k-');
    title(col_lbl{i})
    ylabel('error (s)');
end

%% Save the figures
old_dir = pwd;
cd(data_folder);
saveAllFigs({'png', 'svg'}, [], [fbig, fsmall]); %custom function... saveas should work.
cd(old_dir);

%% Tables with numerical summaries

% First, root-mean-squared error
rmserr = zeros(3, 5);
rmserr(1,1:2) = rms(reshape(permute(win_errs(:, 1:2, :), [2, 3, 1]), 2, []), 2)';
rmserr(1,3:end) = nan;
rmserr(2, :) = rms(reshape(permute(reg_errs, [2 3 1]), 5, []), 2)';
rmserr(3, :) = rms(reshape(permute(map_errs, [2 3 1]), 5, []), 2)';

% Overall RMS Error
rmstbl = array2table(rmserr, 'VariableNames', col_lbl, 'RowNames', mthd_lbl);
fprintf(1, 'RMSE:\n');
disp(rmstbl);

% Next, overall mean error
meanerr = zeros(3, 5);
meanerr(1,1:2) = mean(mean(win_errs(:, 1:2, :), 3), 1);
meanerr(1,3:end) = nan;
meanerr(2, :) = mean(mean(reg_errs, 3), 1);
meanerr(3, :) = mean(mean(map_errs, 3), 1);

meantbl = array2table(meanerr, 'VariableNames', col_lbl, 'RowNames', mthd_lbl);
fprintf(1, 'Mean Error:\n');
disp(meantbl);

%% Write to file
timestamp =  datestr(now, 'yyyy-mm-dd-hh-MM-ss');
writetable(rmstbl, fullfile(data_folder, sprintf('RMSE%s.csv', timestamp)), ...
    'WriteVariableNames', true, 'WriteRowNames', true);
writetable(meantbl, fullfile(data_folder, sprintf('MeanError%s.csv', timestamp)), ...
    'WriteVariableNames', true, 'WriteRowNames', true);