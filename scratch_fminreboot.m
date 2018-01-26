%% fminsearch probs
% fminsearch can be unreliable. One approach is to restart it initialized
% to the previous result.  Let's see how that works.

%% Settings

% stimulation settings
stim_rate = 4; % Stim/s
block_length = 60; % s
inter_block_interval = 10; % s
n_block = 10; % number of blocks
pTar = 0.1; % proportion of stimuli that are targets

MAX_ATTEMPT = 4; % Let MLE try to converge this many times before moving on

% exgaussian RT parameters
%mu = 0.35;
%sigma = 0.1;
%tau = 0.25;
mu = 0.3;
sigma = 0.1;
tau = 0.15;

rtmin = 0;

% Window method settings
win_lo = 0.0;
win_hi = 1.0; % defines start and stop of window.

% exgaussian random numbers
exgr = @(sz) max(normrnd(mu, sigma, sz) + exprnd(tau, sz), ones(sz).*rtmin);

nsim = 100;

conditions = [1.0 .001 mu sigma tau];

iCond = 1;
pHit = conditions(iCond, 1);
pFa = conditions(iCond, 2);
mu = conditions(iCond, 3);
sigma = conditions(iCond, 4);
tau = conditions(iCond, 5);

%% Run the simulation
% Setup stimulus times
block_stim = 0:(1/stim_rate):block_length;
stim_time_mtx = repmat(block_stim(:), 1, n_block);
blk_add = (0:(n_block-1)).*(block_stim(end) + inter_block_interval);
stim_time_mtx = bsxfun(@plus, blk_add, stim_time_mtx);
stim_time = stim_time_mtx(:)';

% setup stimulus labels
nTar = round(numel(stim_time)*pTar);
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

%% Run the MLE
m = rpe.RSVPPerformanceMAP(stim_time, stim_label, button_time);
m.time_resolution = 0.01;
m.use_prior = false;
m.diag = false;

[hr_m, far_m, exit_flag] = m.estimatePerformance();
converged(iCond, iSim) = exit_flag;
map_estimates(iCond, :, iSim) = [hr_m, far_m, m.mu, m.sigma, m.tau];


