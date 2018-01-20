%% Overview
% This script is for testing RSVPPerformanceML2.
% The idea of ML2 is to use MLE to simultaneously estimate the parameters
% of the exgaussian RT-PDF. I have no idea if/how this will work.
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

% True performance parameters
pHit = 0.8; % hit rate
pFa = 0.01; % false alarm rate

% exgaussian RT parameters
mu = 0.35;
sigma = 0.1;
tau = 0.25;

rtmin = 0;

% exgaussian random numbers
exgr = @(sz) max(normrnd(mu, sigma, sz) + exprnd(tau, sz), ones(sz).*rtmin);

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

%% Visualize the experiment timeline
% % Not a great visualization, but might be good for debugging
% figure('Name', 'Simulated Timeline');
% stem(tar_times, ones(size(tar_times)), 'b', 'marker', 'none');
% hold on;
% stem(nt_times, ones(size(nt_times)), 'g', 'marker', 'none');
% stem(button_time, 1.1 * ones(size(button_time)), 'k', 'marker', 'none');
% 

%% Run the MLE2 estimator
% e = rpe.RSVPPerformanceML2(stim_time, stim_label, button_time);
% 
% % reduce time resolution
% e.time_resolution = 0.01;
% t0 = tic;
% [hr_lr, far_lr] = e.runEstimates();
% fprintf(1, 'Low res mle2 took %g s\n', toc(t0));
% 
% fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
%     hr_lr, far_lr, e.mu, e.sigma, e.tau);

%% Run the MLE3 estimator
% e3 = rpe.RSVPPerformanceML3(stim_time, stim_label, button_time);
% e3.time_resolution = 0.01;
% t0 = tic;
% [hr_lr, far_lr] = e3.runEstimates();
% fprintf(1, 'Low res mle3 took %g s\n', toc(t0));
% 
% fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
%     hr_lr, far_lr, e3.mu, e3.sigma, e3.tau);

%% this implementation is faster and stabler than MLE3, above
% run MAP in MLE mode
ml = rpe.RSVPPerformanceMAP(stim_time, stim_label, button_time);
ml.time_resolution = 0.01;
ml.use_prior = false; 
[hr_ml, far_ml] = ml.runEstimates();

% Examine log likelihood of obtained solution:
ll_ml = ml.logLikelihood(hr_ml, far_ml, [ml.mu, ml.sigma, ml.tau], ml.buildTimeIdx());
ll_ml_true = ml.logLikelihood(pHit, pFa, [mu, sigma, tau], ml.buildTimeIdx());

fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
    hr_ml, far_ml, ml.mu, ml.sigma, ml.tau);
fprintf(1, 'Log likelihood of obtained solution: %f\nLog likelihood of true params: %f\n', ...
    ll_ml, ll_ml_true);
fprintf(1, 'Difference (pos: obtained more likely than actual): %f\n', ...
    ll_ml-ll_ml_true);

%% Run the MAP estimator
m = rpe.RSVPPerformanceMAP(stim_time, stim_label, button_time);
m.time_resolution = 0.01;
m.use_prior = true; 
m.diag = true;
[hr_m, far_m] = m.runEstimates();

% plot exgaussian pdf
figure;
x = 0:.01:1.5;
plot(x, [exgaussPdf(x, m.mu, m.sigma, m.tau); exgaussPdf(x, mu, sigma, tau)]);
legend({'estimate', 'true'});
hold on;
histogram(hit_responses-hit_times, 'Normalization', 'pdf');
title('MAP RTPDF');
% Examine log likelihood of obtained solution:
ll_m = ml.logLikelihood(hr_m, far_m, [m.mu, m.sigma, m.tau], m.buildTimeIdx());
ll_m_true = m.logLikelihood(pHit, pFa, [mu, sigma, tau], m.buildTimeIdx());

fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
    hr_m, far_m, m.mu, m.sigma, m.tau);
fprintf(1, 'Log likelihood of obtained solution: %f\nLog likelihood of true params: %f\n', ...
    ll_m, ll_m_true);
fprintf(1, 'Difference (pos: obtained more likely than actual): %f\n', ...
    ll_m-ll_m_true);

%% Run the bayesian estimator
% b = rpe.RSVPPerformanceBayes(stim_time, stim_label, button_time);
% b.time_resolution = 0.1;
% b.diag = true;
% t0 = tic;
% [hr_b, far_b] = b.runEstimates();
% fprintf(1, 'Bayes took %g s\n', toc(t0));
% 
% fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
%     hr_b, far_b, b.mu, b.sigma, b.tau);

% Results: many orders of magnitude slower

% Copyright notice
%    Copyright 2017 Benjamin T. Files
% 
%    Licensed under the Apache License, Version 2.0 (the "License");
%    you may not use this file except in compliance with the License.
%    You may obtain a copy of the License at
% 
%        http://www.apache.org/licenses/LICENSE-2.0
% 
%    Unless required by applicable law or agreed to in writing, software
%    distributed under the License is distributed on an "AS IS" BASIS,
%    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
%    implied. See the License for the specific language governing
%    permissions and limitations under the License.