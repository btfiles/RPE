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
% This script runs a simulation with one set of parameters. It collects
% results from four estimation methods: Window, Regression, MLE/heuristic,
% MLE/simultaneous. It may serve as a basis for parameter exploration and
% repeated simulations.

s_condition = [.6, .01]; % [HR, FAR]
s_do_figures = true; % produce figures at the conclusion of the simulation?
s_nsim = 10; % Repeat the simulation this many times to summarize results.

% rng('default') % will get you the same results each time.
% rng('shuffle') % will get you different results each time.

% stimulation settings
% The experiment is organized as a series of blocks with breaks between
% blocks.
s_stim_rate = 4; % Stim/s
s_block_length = 60; % s
s_inter_block_interval = 10; % s
s_n_block = 10; % number of blocks
s_pTar = 0.1; % proportion of stimuli that are targets

% exgaussian RT parameters (s)
s_mu = 0.3;
s_sigma = 0.1;
s_tau = 0.15;

% time resolution of the regression & mle estimators. All of these methods
% discretize time, so this sets the size of the discretization.
s_time_res = 0.01; % s
% 0.01 seems about as good as 0.001 and dramatically reduces runtime.

% Estimation method identifiers
% short identifiers for the estimation methods
WIN_IDX = 1;
REG_IDX = 2;
MLE_IDX = 3;
MLE2_IDX = 4;

s_method_names = {'win', 'reg', 'mle', 'mle2'};
s_keep_rt = [0, 1, 0, 1]; %

% Note: changing the above identifiers will only result in cosmetic changes
% and/or break things, particularly the plotting at the end. It does not
% actually control what methods are used.

% Window method settings
win_lo = 0.0;
win_hi = 1.0; % defines start and stop of window.
% Other method settings are stored as default member variable values.

%% values derived from settings
% exgaussian random numbers
exgr = @(sz) normrnd(s_mu, s_sigma, sz) + exprnd(s_tau, sz);

n_method = numel(s_method_names);
n_rtm = sum(s_keep_rt); % number of rt methods kept

%% Initialize trackers
sim_results_hr = zeros(s_nsim, n_method);
sim_results_far = zeros(s_nsim, n_method);
sim_times = zeros(s_nsim, n_method);
rt_params = zeros(s_nsim, 3, n_rtm); % For the 3 parameters of the rt pdf

% True performance parameters
pHit = s_condition(1); % hit rate
pFa = s_condition(2); % false alarm rate
fprintf(1, '============ Starting simulation: HR: %f, FAR: %f ============\n', ...
    pHit, pFa);
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
    
    % randomly select nHit targets to get hit responses
    hit_idx = false(size(tar_times));
    hit_idx(1:nHit) = true;
    hit_idx = hit_idx(randperm(numel(hit_idx)));
    hit_times = tar_times(hit_idx);
    
    % draw from the rt distribution enough values for each hit
    hit_responses = exgr(size(hit_times)) + hit_times;
    
    % now with false alarms
    nFa = round(pFa*sum(~stim_label));
    nt_times = stim_time(~stim_label);
    
    % randomly select nFa nontargets to get false alarms
    fa_idx = false(size(nt_times));
    fa_idx(1:nFa) = true;
    fa_idx = fa_idx(randperm(numel(fa_idx)));
    fa_times = nt_times(fa_idx);
    
    % draw from the rt distribution
    fa_responses = exgr(size(fa_times)) + fa_times;
    
    % Here, we obliterate the distinction between hits and false alarms,
    % because in reality we just know if the button was pressed.
    button_time = sort([hit_responses fa_responses]);
    
    % It might be worth saving button_time, stim_time, and stim_label to
    % exactly reproduce these simulations, although using a particular rng
    % seed ought to accomplish something similar.
    
    %% Conventional window analysis
    % The idea of the window analysis is that any response that falls
    % within a window of time after a target is a hit, while all others are
    % a false alarm. It gets a little complicated when multiple responses
    % fall within a window and/or the windows of multiple target stimuli
    % overlap.
    
    fprintf(1, 'Window analysis %d\n', iSim);
    t0 = tic;
    claimed_hit = false(size(button_time)); % track if a response has been claimed by a target image

    for iTar = 1:numel(tar_times)
        tt = tar_times(iTar);
        
        % find responses that fall within the window of time following this
        % target
        in_win = button_time > tt + win_lo & button_time < tt+win_hi;
        
        % What if there are multiple responses in the window?
        % Claim the first one that has not previously been claimed. Leave
        % the rest as either false alarms or hits to later targets.
        if nnz(in_win & ~claimed_hit)>1
            in_win_idx = find(in_win & ~claimed_hit);
            in_win(in_win_idx(2:end)) = false;
        end
        if any(in_win & ~claimed_hit),
            claimed_hit(in_win & ~claimed_hit) = true;
        end
        
    end
    
    win_hr = sum(claimed_hit)/numel(tar_times);
    win_far = sum(~claimed_hit)/numel(nt_times);
    
    sim_results_hr(iSim, WIN_IDX) = win_hr;
    sim_results_far(iSim, WIN_IDX) = win_far;
    
    twin = toc(t0);
    sim_times(iSim, WIN_IDX) = twin;
    fprintf(1, 'Finished window in %f s\n', twin);
    
    %% Do the regression estimation
    % The regression method uses a heuristic to estimate the response time
    % distribution. Briefly, it assumes any response within a window of
    % time after a target was caused by that target to build a sample of
    % response times and then uses maximum likelihood to estimate the
    % parameters of an exgaussian that were most likely to generate that
    % sample. Armed with this rt distribution, it uses that to predict what
    % stimulus was likely to have generated each response (i.e. each image
    % gets some degree of attribution). The expected attribution of an
    % image depends simultaneously on the HR and FAR of the subject. So, we
    % use OLS regression to estimate hr and far.
    % For details, see Files & Marathe 2016 j. neurosci. meth.
    
    fprintf(1, 'Regression analysis %d\n', iSim);
    t0 = tic;
    reg = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
    reg.time_resolution = s_time_res; % for non-default time res
    [hrr, farr] = reg.runEstimates();
    treg = toc(t0);
    
    sim_results_hr(iSim, REG_IDX) = hrr;
    sim_results_far(iSim, REG_IDX) = farr;
    sim_times(iSim, REG_IDX) = treg;
    fprintf(1, 'Finished regression in %f s\n', treg);
    
    rt_params(iSim, 1:3, sum(s_keep_rt(1:REG_IDX))) = ...
        [reg.mu, reg.sigma, reg.tau];
    
    %% Do the maximum likelihood estimation
    % The maximum likelihood estimator estimates the response time
    % distribution in the same way as the regression estimation. For
    % background, the probability of a response occurring at time T depends
    % on what stimuli preceded T by how much time, the hit rate and false
    % alarm rate and the rt distrubtion. So, if we select an rt
    % distribution and a hr and far, we can compute the probability a
    % response occurred at time T. If we consider the results of the
    % experiment as a series of time bins into which a response either did
    % or did not fall, we can compute the probability of a given result as
    % the joint probability of obtaining responses in all bins that
    % responses were collected and of not obtaining responses when no
    % responses were collected. Here, an iterative optimizer is used to
    % find the hr and far such that this joint probability is maximized.
    
    fprintf(1, 'Maximum Likelihood heuristic estimation %d\n', iSim);
    t0 = tic;
    ml = rpe.RSVPPerformanceML(stim_time, stim_label, button_time);
    ml.time_resolution = s_time_res;
    
    % Now run the estimator
    [hr, far] = ml.runEstimates();
    tml = toc(t0);
    sim_results_hr(iSim, MLE_IDX) = hr;
    sim_results_far(iSim, MLE_IDX) = far;
    sim_times(iSim, MLE_IDX) = tml;
    fprintf(1, 'Finished mle in %f s\n', tml);
    
    
    %% Do the maximum likelihood estimation 2
    % This is similar to the maximum likelihood estimator above, but
    % instead of estimating the parameters of the rt distribution
    % separately, we estimate them simultaneously with the hr and far.
    
    fprintf(1, 'Maximum Likelihood simultaneous estimation %d\n', iSim);
    t0 = tic;
    e2 = rpe.RSVPPerformanceML2(stim_time, stim_label, button_time);
    e2.time_resolution = s_time_res;
    
    % Now run the estimator
    [hr2, far2] = e2.runEstimates();
    tml2 = toc(t0);
    sim_results_hr(iSim, MLE2_IDX) = hr2;
    sim_results_far(iSim, MLE2_IDX) = far2;
    
    sim_times(iSim, MLE2_IDX) = tml2;
    fprintf(1, 'Finished mle 2 in %f s\n', tml2);
    rt_params(iSim, 1:3, sum(s_keep_rt(1:MLE2_IDX))) = [e2.mu, e2.sigma, e2.tau];
end

%%
if ~s_do_figures
    return
end
%% Boxplot of estimated HR and FAR by all methods

figure('name', 'EstimateBoxplots');
subplot(1,2,1);
boxplot(sim_results_hr, s_method_names);
ylabel('hr');
hold on;
plot(xlim, s_condition(1).*[1 1], 'k--');
subplot(1,2,2);
boxplot(sim_results_far, s_method_names);
ylabel('far');
hold on;
plot(xlim, s_condition(2).*[1 1], 'k--');
drawnow;
pause(.2);

%% summary of rmse HR and FAR
tmp = repmat(s_condition, n_method, 1);
err = bsxfun(@minus, [sim_results_hr, sim_results_far], tmp(:)');
rmse = squeeze(sqrt(mean(err.^2, 1)));

bsf = @(d) sqrt(mean(d));
t0 = tic;
%on my machine, parallelizing bootstrapping is not worth the overhead
ci = bootci(10000, {bsf, err.^2}, 'Options', struct('UserParallel', false)); 
fprintf(1, 'Bootstrapping RMSE CI took %f\n', toc(t0));

L = rmse - ci(1, :);
U = ci(2, :) - rmse;

%% Produce the figure
figure('name', 'rmse overview')
% HR panel
subplot(1,2,1)
ebx = 1:numel(s_method_names);
r = rmse(1:n_method);
l = L(1:n_method);
u = U(1:n_method);
errorbar(ebx, r, l, u, '.', 'linewidth', 1);

ylabel('HR RMSE');
set(gca, 'XTick', ebx, 'XTickLabel', s_method_names);
xlabel('estimator');

hold on;
% Put reference line for the lowest error method
[~, mnidx] = min(r);
plot(xlim(), r(mnidx).*[1 1], 'r-', 'linewidth', 0.5);
plot([xlim()' xlim()'], repmat([r(mnidx)+u(mnidx), r(mnidx)-l(mnidx)], 2, 1), 'k-', 'linewidth', 0.5);

% FAR panel
subplot(1,2,2)
idx = n_method + (1:n_method);
r = rmse(idx);
l = L(idx);
u = U(idx);
errorbar(ebx, r, l, u, '.', 'linewidth', 1);

ylabel('FAR RMSE');
set(gca, 'XTick', ebx, 'XTickLabel', s_method_names);
xlabel('estimator');

hold on;
[~, mnidx] = min(r);
plot(xlim(), r(mnidx).*[1 1], 'r-', 'linewidth', 0.5);
plot([xlim()' xlim()'], repmat([r(mnidx)+u(mnidx), r(mnidx)-l(mnidx)], 2, 1), 'k-', 'linewidth', 0.5);

%% RT PDF values
rt_true = [s_mu, s_sigma, s_tau];
figure('Name', 'RTOverview')

g1 = repmat({'mu', 'sig', 'tau'}, 1, n_rtm);
c_method = s_method_names(logical(s_keep_rt))';
g2 = repmat(c_method, 1, 3)';

h = boxplot(reshape(rt_params, s_nsim, []), {g1(:), g2(:)}, ...
    'colors', 'bg', 'colorgroup', repmat([1 2], 1, 3),...
    'factorgap', [15, 1], 'factorseparator', 1);

ylabel('Parameter Estimate (s)');

% mark the true parameter values
xc = get(h(strcmp(get(h,'Tag'), 'Upper Whisker')), 'XData');
txm = reshape(cat(1, xc{:})', 4, []);
hold on;
plot(mean(txm), rt_true, 'k.');
hl = plot(txm([1 3], :), repmat(rt_true, 2, 1), 'k-');
legend(hl(1), 'True value');
drawnow;
pause(0.05);

%% RT RMSE
rt_err2 = bsxfun(@minus, rt_params(:,:), repmat(rt_true, 1, n_rtm)).^2;
rt_rmse = sqrt(mean(rt_err2, 1));

bsf = @(d) sqrt(mean(d));
tic
rt_ci = bootci(10000, {bsf, rt_err2}, 'Options', struct('UserParallel', false));
toc

L = rt_rmse - squeeze(rt_ci(1, :, :));
U = squeeze(rt_ci(2, :, :)) - rt_rmse;
%% plot RT RMSE
x = bsxfun(@plus, repmat(1:3, n_rtm, 1)', [-.1 .1]);
rs = @(d) reshape(d,3,n_rtm);
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