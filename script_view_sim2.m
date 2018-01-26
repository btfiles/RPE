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

%%
for jCol = 1:5
    figure('name', col_lbl{jCol});
    
    d = {win_estimates, reg_estimates};
    axl = zeros(1,2);
    for i = 1:2
        subplot(3,1,i)
        ax = gca;
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
    
end