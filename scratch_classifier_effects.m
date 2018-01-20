%% Overview
% This script is a first stab at determining if the better RT PDF estimate
% from the "mle2" method actually has consequences for button-based image
% classification.

%% Settings
s_condition = [.6, .1]; % [HR, FAR]
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
s_sigma = 0.2;
s_tau = 0.15;

% time resolution of the regression & mle estimators. All of these methods
% discretize time, so this sets the size of the discretization.
s_time_res = 0.01; % s
% 0.01 seems about as good as 0.001 and dramatically reduces runtime.
%% values derived from settings
% exgaussian random numbers
exgr = @(sz) normrnd(s_mu, s_sigma, sz) + exprnd(s_tau, sz);

%% Run the simulation

% True performance parameters
pHit = s_condition(1); % hit rate
pFa = s_condition(2); % false alarm rate

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

%% get the RT PDF estimates from regression and MLE2 methods
reg = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
reg.time_resolution = s_time_res; % for non-default time res
reg.estimatePdf();

t0 = tic;
e2 = rpe.RSVPPerformanceML2(stim_time, stim_label, button_time);
e2.time_resolution = s_time_res;

% Now run the estimator
[hr2, far2] = e2.runEstimates();
tml2 = toc(t0);
fprintf(1, 'MLE2 done, took %f s.\n', tml2);

%% Apply button classifier using the two pdfs
t0 = tic;
score_reg = buttonClassifier(stim_time, button_time, reg.pdf_fcn, reg.pdf_support);
score_mle = buttonClassifier(stim_time, button_time, e2.pdf_fcn, e2.pdf_support);
time_score = toc(t0);
fprintf(1, 'Scoring took %f s.\n', time_score);

%% Visualize results
[x_reg, y_reg, ~, az_reg] = perfcurve(stim_label, score_reg, true);
[x_mle, y_mle, ~, az_mle] = perfcurve(stim_label, score_mle, true);

figure;
hold on;
plot(x_reg, y_reg);
plot(x_mle, y_mle);
xlabel('False positive rate');
ylabel('True positive rate');
legend({['Reg Az: ' num2str(az_reg)], ['MLE Az: ' num2str(az_mle)]}, ...
    'Location', 'SouthEast');

axis square;