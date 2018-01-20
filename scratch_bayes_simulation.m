%% investigating a more bayesian estimator
%
%  Now that I know about using MCMC to sample the posterior, it seems like
%  I should be able to use that to make a much better estimator.
%

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

% True performance parameters
pHit = 0.6; % hit rate
pFa = 0.01; % false alarm rate

% exgaussian RT parameters
mu = 0.3;
sigma = 0.05;
tau = 0.1;

% exgaussian random numbers
exgr = @(sz) normrnd(mu, sigma, sz) + exprnd(tau, sz);

% analysis parameters
t_res = 0.01;
t_support = 1.5;

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

%% RT Priors

%Hyper-parameters
hp_mu = [-1.5, 0.4];
hp_sig = [-1.5, 0.3];
hp_tau = [-2.5, 1];

% pdfs
pr_mu = @(mu) lognpdf(mu, hp_mu(1), hp_mu(2));
pr_sig = @(sig) lognpdf(sig, hp_sig(1), hp_sig(2));
pr_tau = @(tau) lognpdf(tau, hp_tau(1), hp_tau(2));

%% look at these choices...
t = 0:.01:1.5;
figure;
subplot(2,1,1);
hold on;
ax = gca;
h(1) = plot(ax, t, pr_mu(t));
h(2) = plot(ax, t, pr_sig(t));
h(3) = plot(ax, t, pr_tau(t), '--');
title(ax, 'priors');
legend(ax, h, {'\mu', '\sigma', '\tau'});

subplot(2,1,2);
hold on;
ax = gca;
ndraw = 100;
th = zeros(ndraw, 3);
for i = 1:ndraw
    mu = lognrnd(hp_mu(1), hp_mu(2));
    sig = lognrnd(hp_sig(1), hp_sig(2));
    tau = lognrnd(hp_tau(1), hp_tau(2));
    plot(ax, t, exgaussPdf(t, mu, sig, tau));
    th(i, :) = [mu, sig, tau];
end
title(ax, 'Permitted PDFs');

%% HR prior
% Hyper-parameters
hp_h = [2 .65];

% pdf
pr_h = @(h) betapdf(h, hp_h(1)*hp_h(2), hp_h(1)*(1-hp_h(2)));

%% Full prior
prior = @(th) prod([pr_h(th(1)), pr_mu(th(2)), pr_sig(th(3)), pr_tau(th(4))]);
lik = @(th) rsvpLikelihood(stim_time, stim_label, button_time, t_res, t_support, th(1), th(2:end));

%% Test the likelihood function:
% lik([pHit, mu, sigma, tau]);

% That likelihood function works, but I implemented it much faster in RSVPPerformanceML3
% TODO: Use the RSVPPerformanceML3 implementation with MWG