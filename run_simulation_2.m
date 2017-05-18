%% Simulation settings
% This simulates an RSVP experiment.  Stimuli are shown in blocks, with
% some amount of time between blocks.  Stimulation rate, block length,
% inter-block interval and number of blocks are all configurable.
%
% Some stimuli are assumed to be non-targets, and others are
% targets. The proportion of targets is configurable.
%
% For responses, a proportion of targets (pHit) generate a response and a
% proportion of nontargets (pFa) also generate a response. Response
% latencies are sampled from an exGaussian distribution with configurable
% parameters.
%
% To get a sense of the variability in the ML method vs the Regression
% method, I'm going to run the same simulation 100 times.
outfile = 'hrfarsim_mle2.mat';
s_nsim = 100; % per condition

s_conditions = [.6, .01];

% change to
rng('default')
% if you want to get a sense of how much variability you can get on
% repeated runs.

% stimulation settings
s_stim_rate = 4; % Stim/s
s_block_length = 60; % s
s_inter_block_interval = 10; % s
s_n_block = 10; % number of blocks
s_pTar = 0.1; % proportion of stimuli that are targets

% exgaussian RT parameters
s_mu = 0.3;
s_sigma = 0.1;
s_tau = 0.15;

% time resolution of the whole business
time_res = 0.01; % seconds

method_names = {'win', 'reg', 'mle', 'mle2'};
%% values derived from settings
% exgaussian random numbers
exgr = @(sz) normrnd(s_mu, s_sigma, sz) + exprnd(s_tau, sz);

% number of conditions to simulate
n_cond = size(s_conditions, 1);


%% Initialize trackers
sim_results = zeros(s_nsim, 8, n_cond); % wHr, wFar, rHr, rFar, mlHr, mlFar
sim_times = zeros(s_nsim, 4, n_cond);
rt_params = zeros(s_nsim, 6, n_cond);
for iCond = 1:n_cond
    
    % True performance parameters
    pHit = s_conditions(iCond, 1); % hit rate
    pFa = s_conditions(iCond, 2); % false alarm rate
    fprintf(1, '============ Starting condition %d: HR: %f, FAR: %f ============\n', ...
        iCond, pHit, pFa);
    %%
    for iSim = 1:s_nsim
        %% Run the simulation
        % Setup stimulus times
        block_stim = 0:(1/s_stim_rate):s_block_length;
        stim_time_mtx = repmat(block_stim(:), 1, s_n_block);
        blk_add = (0:(s_n_block-1)).*(block_stim(end) + s_inter_block_interval);
        stim_time_mtx = bsxfun(@plus, blk_add, stim_time_mtx);
        stim_time = stim_time_mtx(:)';
        
        % setup stimulus labels
        nTar = round(numel(stim_time)*s_pTar);
        lbl = false(size(stim_time));
        lbl(1:nTar) = true;
        stim_label = lbl(randperm(numel(stim_time)));
        
        % setup buttonpresses
        nHit = round(pHit*sum(stim_label));
        tar_times = stim_time(stim_label);
        hit_idx = false(size(tar_times));
        hit_idx(1:nHit) = true;
        hit_idx = hit_idx(randperm(numel(hit_idx)));
        hit_times = tar_times(hit_idx);
        hit_responses = exgr(size(hit_times)) + hit_times;
        
        nFa = round(pFa*sum(~stim_label));
        nt_times = stim_time(~stim_label);
        fa_idx = false(size(nt_times));
        fa_idx(1:nFa) = true;
        fa_idx = fa_idx(randperm(numel(fa_idx)));
        fa_times = nt_times(fa_idx);
        fa_responses = exgr(size(fa_times)) + fa_times;
        
        button_time = sort([hit_responses fa_responses]);
        
        %% Conventional window analysis
        fprintf(1, 'Window analysis %d\n', iSim);
        t0 = tic;
        win_lo = 0.0;
        win_hi = 1.0;
        in_any_win = false(size(button_time));
        n_hit = 0;
        for iTar = 1:numel(tar_times)
            tt = tar_times(iTar);
            in_win = button_time > tt + win_lo & button_time < tt+win_hi;
            if any(in_win),
                n_hit = n_hit + 1;
            end
            in_any_win(in_win) = true;
        end
        
        win_hr = n_hit/numel(tar_times);
        win_far = sum(~in_any_win)/numel(nt_times);
        
        sim_results(iSim, 1:2, iCond) = [win_hr, win_far];
        twin = toc(t0);
        sim_times(iSim, 1, iCond) = twin;
        fprintf(1, 'Finished window in %f s\n', twin);
        
        %% Do the regression estimation
        fprintf(1, 'Regression analysis %d\n', iSim);
        t0 = tic;
        f = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
        f.time_resolution = time_res;
        [hrr, farr] = f.runEstimates();
        treg = toc(t0);
        
        sim_results(iSim, 3:4, iCond) = [hrr, farr];
        sim_times(iSim, 2, iCond) = treg;
        fprintf(1, 'Finished regression in %f s\n', treg);
        
        rt_params(iSim, 1:3, iCond) = [f.mu, f.sigma, f.tau];
        %% Do the maximum likelihood estimation
        % setup the estimator
        % clear e
        fprintf(1, 'Maximum Likelihood estimation %d\n', iSim);
        t0 = tic;
        e = rpe.RSVPPerformanceML(stim_time, stim_label, button_time);
        e.time_resolution = time_res;
        
        % Now run the estimator
        [hr, far] = e.runEstimates();
        tml = toc(t0);
        sim_results(iSim, 5:6, iCond) = [hr, far];
        sim_times(iSim, 3, iCond) = tml;
        fprintf(1, 'Finished mle in %f s\n', tml);
        
        
        %% Do the maximum likelihood estimation 2
        % setup the estimator
        % clear e
        fprintf(1, 'Maximum Likelihood estimation %d\n', iSim);
        t0 = tic;
        e2 = rpe.RSVPPerformanceML2(stim_time, stim_label, button_time);
        e2.time_resolution = time_res;
        
        % Now run the estimator
        [hr2, far2] = e2.runEstimates();
        tml2 = toc(t0);
        sim_results(iSim, 7:8, iCond) = [hr2, far2];
        sim_times(iSim, 4, iCond) = tml2;
        fprintf(1, 'Finished mle 2 in %f s\n', tml2);
        rt_params(iSim, 4:6, iCond) = [e2.mu, e2.sigma, e2.tau];
    end
    
    %%
%     figure;
%     subplot(1,2,1);
%     boxplot(squeeze(sim_results(:, 1:2:end, iCond)), method_names);
%     ylabel('hr');
%     hold on;
%     plot(xlim, pHit.*[1 1], 'k--');
%     subplot(1,2,2);
%     boxplot(squeeze(sim_results(:, 2:2:end, iCond)), method_names);
%     ylabel('far');
%     hold on;
%     plot(xlim, pFa.*[1 1], 'k--');
%     drawnow
%     pause(.2);
end
%%
save(outfile, 'sim_results', 'sim_times', 'rt_params', 's_*');
%%
for iCond = 1:size(s_conditions,1)
    pHit = s_conditions(iCond, 1);
    pFa = s_conditions(iCond, 2);
    %%
    figure('name', sprintf('Condition%dBoxplots', iCond));
    subplot(1,2,1);
    boxplot(squeeze(sim_results(:, 1:2:end, iCond)), method_names);
    ylabel('hr');
    hold on;
    plot(xlim, pHit.*[1 1], 'k--');
    subplot(1,2,2);
    boxplot(squeeze(sim_results(:, 2:2:end, iCond)), method_names);
    ylabel('far');
    hold on;
    plot(xlim, pFa.*[1 1], 'k--');
    drawnow;
    pause(.2);
end

%% summary of rmse
sz = size(s_conditions, 1);
tmp = zeros(1, size(sim_results,2), sz);
for i = 1:sz,
    tmp(1, 1:2:end, i) = s_conditions(i, 1);
    tmp(1, 2:2:end, i) = s_conditions(i, 2);
end
err = bsxfun(@minus, sim_results, tmp);
rmse = squeeze(sqrt(mean(err.^2, 1)));

err_rs = reshape(err, size(err,1), []);
bsf = @(d) sqrt(mean(d));
tic
ci_rs = bootci(10000, {bsf, err_rs.^2}, 'Options', struct('UserParallel', false));
toc

ci = reshape(ci_rs, 2, size(err,2), size(err,3));

L = rmse - squeeze(ci(1, :, :));
U = squeeze(ci(2, :, :)) - rmse;

%% Show summary
if n_cond==1
    figure('name', 'rmse overview')
    subplot(1,2,1)
    ebx = 1:numel(method_names);
    r = rmse(1:2:end);
    l = L(1:2:end);
    u = U(1:2:end);
    errorbar(ebx, r, l, u, '.', 'linewidth', 1);
    
    ylabel('HR RMSE');
    set(gca, 'XTick', ebx, 'XTickLabel', method_names);
    xlabel('estimator');
    
    hold on;
    [~, mnidx] = min(r);
    plot(xlim(), r(mnidx).*[1 1], 'r-', 'linewidth', 0.5);
    plot([xlim()' xlim()'], repmat([r(mnidx)+u(mnidx), r(mnidx)-l(mnidx)], 2, 1), 'k-', 'linewidth', 0.5);
    
    subplot(1,2,2)
    
    r = rmse(2:2:end);
    l = L(2:2:end);
    u = U(2:2:end);
    errorbar(ebx, r, l, u, '.', 'linewidth', 1);
    
    ylabel('FAR RMSE');
    set(gca, 'XTick', ebx, 'XTickLabel', method_names);
    xlabel('estimator');
    
    hold on;
    [~, mnidx] = min(r);
    plot(xlim(), r(mnidx).*[1 1], 'r-', 'linewidth', 0.5);
    plot([xlim()' xlim()'], repmat([r(mnidx)+u(mnidx), r(mnidx)-l(mnidx)], 2, 1), 'k-', 'linewidth', 0.5);
else
    xtl = sprintf('H: %1.3f,F: %1.3f|', s_conditions');
    xtl = strsplit(xtl(1:(end-1)), '|');
    
    figure('name', 'overview', 'position', [996   790   570   689]);
    subplot(2,1,1);
    errorbar(repmat(1:sz, size(rmse,1)/2, 1)', rmse(1:2:end, :)', L(1:2:end, :)', U(1:2:end, :)', '.-');
    ylabel('RMSE');
    xlim([0, sz+1]);
    set(gca, 'xtick', (1:sz), 'XTickLabel', xtl, 'XTickLabelRotation', 45);
    xlabel('condition');
    drawnow;
    pause(.05);
    legend({'win hr', 'reg hr', 'mle hr'}, 'location', 'best');
    subplot(2,1,2);
    errorbar(repmat(1:sz, size(rmse,1)/2, 1)', rmse(2:2:end, :)', L(2:2:end, :)', U(2:2:end, :)', '.-');
    ylabel('RMSE');
    xlim([0, sz+1]);
    set(gca, 'xtick', (1:sz), 'XTickLabel', xtl, 'XTickLabelRotation', 45);
    xlabel('condition');
    drawnow;
    pause(.05);
    legend({'win far', 'reg far', 'mle far'}, 'location', 'best');
end
drawnow;
pause(0.05);
%% RT PDF values
rt_true = [s_mu, s_sigma, s_tau];
figure('Name', 'RTOverview')

g1 = repmat({'mu', 'sig', 'tau'}, 1, 2);
g2 = [repmat({'mle'}, 1, 3), repmat({'mle2'}, 1, 3)];
h = boxplot(rt_params, {g1, g2}, ...
    'colors', 'bg', 'colorgroup', repmat([1 2], 1, 3),...
    'factorgap', [15, 1], 'factorseparator', 1);

ylabel('Parameter Estimate (s)');

xc = get(h(strcmp(get(h,'Tag'), 'Upper Whisker')), 'XData');
txm = reshape(cat(1, xc{:})', 4, []);
hold on;
plot(mean(txm), rt_true, 'k.');
hl = plot(txm([1 3], :), repmat(rt_true, 2, 1), 'k-');
legend(hl(1), 'True value');
drawnow;
pause(0.05);
%% RT RMSE
rt_err2 = bsxfun(@minus, rt_params, repmat(rt_true, 1, 2)).^2;

rt_rmse = sqrt(mean(rt_err2, 1));

bsf = @(d) sqrt(mean(d));
tic
rt_ci = bootci(10000, {bsf, rt_err2}, 'Options', struct('UserParallel', false));
toc

L = rt_rmse - squeeze(rt_ci(1, :, :));
U = squeeze(rt_ci(2, :, :)) - rt_rmse;
%%
x = bsxfun(@plus, repmat(1:3, numel(rt_rmse)/3, 1)', [-.1 .1]);
rs = @(d) reshape(d,3,numel(rt_rmse)/3);
y = rs(rt_rmse);
l = rs(L);
u = rs(U);

figure('name', 'RTRMSE');
errorbar(x, y, l, u, '.')
ax = gca;
ax.XTick = 1:3;
ax.XTickLabel = {'mu', 'sigma', 'tau'};
ylabel('RMSE (s)');
legend({'MLE', 'MLE2'}, 'location', 'best');