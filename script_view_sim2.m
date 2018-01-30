%% Quick visualizations of Jonroy's small simulations

mthd_lbl = {'win', 'reg', 'mle'};
col_lbl = {'HR', 'FAR', 'Mu', 'Sigma', 'Tau'};
% load('MLE Sim 2 Data/MLE Sim 2 results workspace.mat');
% close all force
%%
% for jCol = 1:5
%     figure;
%     d = {win_estimates, reg_estimates, map_estimates};
%
%
%     axl = zeros(1,3);
%     for i = 1:3
%         subplot(3,1,i)
%         simdata = squeeze(d{i}(:, jCol, :))';
%         boxplot(simdata);
%         hold on;
%         plot(1:size(conditions, 1), conditions(:, jCol), 'ko');
%         plot(1:size(conditions, 1), mean(simdata), 'g*');
%         ylabel([col_lbl{jCol} ' ' mthd_lbl{i}]);
%         axl(i) = gca;
%     end
%     linkaxes(axl);
% end

%% It looks like many (10%-15%) of the MLE results don't converge

cvg_threshold = 0; % difference between likelihood of estimate and true should be greater than this

lld = ll_estimate-ll_true;

% positive lld means the estimate has higher likelihood, negative means the
% true value has higher likelihood.
%
% When MLE works, it should find some combination of parameters that have
% at least as high likelihood as the true parameters.

ran_sim = ~all(ll_estimate==0, 2); % some didn't run...

f = figure('Name', 'Convergence');
subplot(1,2,1);
ax = gca;
ax.Parent = f;
histogram(ax, lld(ran_sim,:));
xlabel('likelihood difference');
title('all results');

subplot(1,2,2);
ax = gca;
ax.Parent = f;
histogram(ax, lld(ran_sim & lld>=0));
title('results that apparently converged.');

okll = lld>=0;

%% print proportion of good solutions
fprintf(1, '%2.1f%% of solutions were bad.\n', 100*mean(lld(:)<0));

%%
drawnow
pause(0.1);
for jCol = 1:5
    f = figure('name', col_lbl{jCol});
    
    d = {win_estimates, reg_estimates};
    axl = zeros(1,2);
    for i = 1:2
        subplot(3,1,i)
        ax = gca;
        ax.Parent = f;
        simdata = squeeze(d{i}(ran_sim, jCol, :))';
        boxplot(ax, simdata);
        hold on;
        plot(ax, 1:sum(ran_sim), conditions(ran_sim, jCol), 'ko');
        plot(ax, 1:sum(ran_sim), mean(simdata), 'g*');
        ylabel(ax, [col_lbl{jCol} ' ' mthd_lbl{i}]);
        axl(i) = ax;
    end
    
    subplot(3,1,3);
    hold on;
    ax = gca;
    ax.Parent = f;
    [simdatac, idxc] = deal(cell(25, 1));
    for iCond = 1:25
        if ~ran_sim(iCond)
            continue
        end
        simdatac{iCond} = squeeze(map_estimates(iCond, jCol, okll(iCond, :)))';
        idxc{iCond} = ones(size(simdatac{iCond})).*iCond;
    end
    boxplot(ax, cat(2, simdatac{:}), cat(2, idxc{:}));
    ylabel(ax, [col_lbl{jCol} ' MLE']);
    plot(ax, conditions(ran_sim, jCol), 'ko');
    plot(ax, cellfun(@mean, simdatac(ran_sim)), 'g*');
    title(ax, col_lbl{jCol});
    
    if jCol > 2
        linkaxes([axl(end) ax]);
    else
        linkaxes([axl ax]);
    end
    
    drawnow
    pause(0.1);
end

%% boxplot errors?

figure('Position', [2 28 1278 1487]);
clrs = [82, 43, 114; 78, 146, 49; 170, 127, 57]./255;

lims = [-0.1 0.15; -0.02 0.01; -0.2 0.2; -0.2 0.2; -0.2 0.2];
axl = zeros(1,5);
for i = 1:2
    map_errs = map_estimates-conditions;
    p_map_errs = squeeze(map_errs(:, i, :))';
    reg_errs = reg_estimates-conditions;
    p_reg_errs = squeeze(reg_errs(:, i, :))';
    win_errs = win_estimates-conditions;
    p_win_errs = squeeze(win_errs(:, i, :))';
    
    
    X = zeros(size(p_map_errs, 1), size(p_map_errs, 2) + size(p_reg_errs, 2) + size(p_win_errs, 2));
    
    X(:, 1:3:end) = p_win_errs;
    X(:, 2:3:end) = p_reg_errs;
    X(:, 3:3:end) = p_map_errs;
    
    Gmethod = repmat([1 2 3],1, size(X, 2)/3);
    
    Gcond = reshape(repmat(1:25, 3, 1), 1, []);
    subplot(5,1,i);
    axl(i) = gca();
    hold on;
    pos = 1:100;
    pos(4:4:end) = [];
    bp = boxplot(X, {Gcond, Gmethod}, 'plotstyle', 'traditional', 'symbol', '.', ...
        'medianstyle', 'line', 'colors', clrs, 'datalim', lims(i,:), ...
        'notch', 'on', 'positions', pos);
    plot(xlim(), [0 0], 'k-');
    
    set(bp(1:2, :), 'LineStyle', '-');
    
    % Work out ticks and labels
    %     tickxC = get(bp(1, :), 'XData');
    %     tickx = unique(cat(2, tickxC{:}));
    
    tickx = 2:4:100;
    set(gca, 'XTick', tickx, 'XTickLabel', ' ');
    
    legend(bp(5, 1:3), mthd_lbl, 'location', 'best');
    
    ylabel([col_lbl{i} ' error']);
end

for i = 3:5
    map_errs = map_estimates-conditions;
    p_map_errs = squeeze(map_errs(:, i, :))';
    reg_errs = reg_estimates-conditions;
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
    
    
    %     % Work out ticks and labels
    tickx = 2:4:100;
    set(gca, 'XTick', tickx, 'XTickLabel', ' ');
    
    legend(bp(5, 1:2), mthd_lbl(2:end));
    
    ylabel([col_lbl{i} ' error']);
end
xtl = sprintf('(%2.1f, %2.1f)\n', conditions(:,1:2)'.*100);
set(gca, 'XTick', tickx, 'XTickLabel', xtl, 'XTickLabelRotation', 45);
linkaxes(axl, 'x');
xlabel('Simulated Condition (HR, FAR)');
%%
f = figure('name', 'RMSE');

errs = {win_estimates(:, 1:2,:) - conditions(:, 1:2), reg_estimates - conditions, map_estimates - conditions};
axl = zeros(size(errs));
for i = 1:numel(errs)
    subplot(1,3,i);
    ax = gca;
    ax.Parent = f;
    rmsc = sqrt(mean(errs{i}.^2, 3));
    imagesc(ax, rmsc);
    title(ax, mthd_lbl{i});
    colorbar(ax);
    axl(i) = ax;
end
linkprop(axl, 'CLim');

%%
f = figure('name', 'RMSEScatter');
rmsc = cell(size(errs));
for j = 1:3
    rmsc{j} = sqrt(mean(errs{j}.^2, 3));
end
for i = 1:5
    subplot(5, 1, i);
    hold on;
    idx = cellfun(@(r) size(r,2)>= i, rmsc);
    cellfun(@(r) plot(r(:, i)), rmsc(idx));
    legend(mthd_lbl(idx));
    ylabel(col_lbl{i});
end
