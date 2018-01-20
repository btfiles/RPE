%% Overview
% This script is for running a few simulations to compare MAP vs
% Regression.
%
% The bulk of the script simulates an RSVP target detection experiment.  To
% use this code with your data, you need to setup the following three
% variables:
% stim_time:    nstim-length vector of stimulus times, in seconds with
%               whatever precision is available, at least millisecond.
% stim_label:   nstim-length vector of stimulus labels, true for target,
%               false otherwise.
% button_time:  The time (again, in seconds) of button press starts.
%
%
% See Also rpe.RSVPPerformanceEstimator rpe.exGaussPdf rpe.fitExGauss
%
% Reference
% Files, B. T., & Marathe, A. R. (2016). A regression method for estimating
% performance in a rapid serial visual presentation target detection task.
% Journal of Neuroscience Methods, 258, 114?123.
% http://doi.org/10.1016/j.jneumeth.2015.11.003

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

% for deterministic performance
%rng('default');

% change to
rng('shuffle')
% if you want to get a sense of how much variability you can get on
% repeated runs.

% stimulation settings
stim_rate = 4; % Stim/s
block_length = 60; % s
inter_block_interval = 10; % s
n_block = 10; % number of blocks
pTar = 0.1; % proportion of stimuli that are targets

% exgaussian RT parameters
mu = 0.35;
sigma = 0.1;
tau = 0.25;

rtmin = 0;

% exgaussian random numbers
exgr = @(sz) max(normrnd(mu, sigma, sz) + exprnd(tau, sz), ones(sz).*rtmin);

nsim = 10;

conditions = ...
    [.6 .1 mu sigma tau;
    .6 .05 mu sigma tau;
    .6 .01 mu sigma tau;
    .6 .005 mu sigma tau;
    .9 .1 mu sigma tau;
    .95 .1 mu sigma tau;
    .99 .1 mu sigma tau;
    .995 .1 mu sigma tau];

map_estimates = zeros([size(conditions), nsim]);
reg_estimates = zeros([size(conditions), nsim]);
[ll_true, ll_estimate, reg_t, ml_t] = deal(zeros(size(conditions,1), nsim));
wb = waitbar(0);
simcount = 0;
t0 = tic;
for iCond = 1:size(conditions,1)
    % True performance parameters
    pHit = conditions(iCond, 1);
    pFa = conditions(iCond, 2);
    mu = conditions(iCond, 3);
    sigma = conditions(iCond, 4);
    tau = conditions(iCond, 5);
    
    for iSim = 1:nsim
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
        
        %% Run the regression estimator
        tr = tic;
        e = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
        [hr, far] = e.runEstimates();
        reg_t(iCond, iSim) = toc(tr);
        
        reg_estimates(iCond, :, iSim) = [hr, far, e.mu, e.sigma, e.tau];
        %% Run the MAP estimator
        tm = tic;
        m = rpe.RSVPPerformanceMAP(stim_time, stim_label, button_time);
        m.time_resolution = 0.01;
        m.use_prior = true;
        m.diag = false;
        [hr_m, far_m] = m.runEstimates();
        ml_t(iCond, iSim) = toc(tm);
        map_estimates(iCond, :, iSim) = [hr_m, far_m, m.mu, m.sigma, m.tau];
        
        % plot exgaussian pdf
        % figure;
        % x = 0:.01:1.5;
        % plot(x, [exgaussPdf(x, m.mu, m.sigma, m.tau); exgaussPdf(x, mu, sigma, tau)]);
        % hold on;
        % histogram(hit_responses-hit_times, 'Normalization', 'pdf');
        % legend({'estimated pdf', 'true pdf', 'rt samples'});
        % title('MAP RTPDF');
        
        % Examine log likelihood of obtained solution:
        ll_estimate(iCond, iSim) = m.logLikelihood(hr_m, far_m, [m.mu, m.sigma, m.tau], m.buildTimeIdx());
        ll_true(iCond, iSim) = m.logLikelihood(pHit, pFa, [mu, sigma, tau], m.buildTimeIdx());
        
        % fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
        %     hr_m, far_m, m.mu, m.sigma, m.tau);
        % fprintf(1, 'Log likelihood of obtained solution: %f\nLog likelihood of true params: %f\n', ...
        %     ll_m, ll_m_true);
        % fprintf(1, 'Difference (pos: obtained more likely than actual): %f\n', ...
        %     ll_m-ll_m_true);
        simcount = simcount+1;
        waitbar(simcount/(size(conditions,1)*nsim), wb);
    end
end
toc(t0);