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
pHit = 0.6; % hit rate
pFa = 0.01; % false alarm rate

% exgaussian RT parameters
mu = 0.3;
sigma = 0.05;
tau = 0.1;

% exgaussian random numbers
exgr = @(sz) normrnd(mu, sigma, sz) + exprnd(tau, sz);

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
e = rpe.RSVPPerformanceML2(stim_time, stim_label, button_time);

% reduce time resolution
e.time_resolution = 0.01;
t0 = tic;
[hr_lr, far_lr] = e.runEstimates();
fprintf(1, 'Low res mle2 took %g s\n', toc(t0));

fprintf(1, 'Estimates:\nHR\t%.3f\nFAR\t%.3f\nMU\t%f\nSIG\t%f\nTAU\t%f\n\n', ...
    hr_lr, far_lr, e.mu, e.sigma, e.tau);

%% Visualize the response time distributions
figure('Name', 'Simulated response distribution');
subplot(2,1,1);
edg = 0:.05:1.5;
N = histc([fa_responses-fa_times hit_responses-hit_times], edg);

bar(edg, N, 'histc');
x = 0:.001:1.5;
pdf = rpe.exGaussPdf(x, mu, sigma, tau);
npdf = numel(button_time)*pdf*diff(edg([1 2]));
hold on;
h = plot(x, npdf, 'r');
legend(h, sprintf('True dist: EXG(%.3f, %.3f, %.3f)', mu, sigma, tau));
title('true distribution')

subplot(2,1,2);
eN = histc(e.rt_list, edg);
epdf = e.pdf_fcn(x);
enpdf = numel(e.rt_list)*epdf*diff(edg([1 2]));
bar(edg, eN, 'histc');
hold on;
h(1) = plot(x, enpdf, 'r');
% add the old way distribution:
f = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
f.estimatePdf();
h(2) = plot(x, numel(f.rt_list)*f.pdf_fcn(x)*diff(edg([1 2])), 'g--');

legend(h, {sprintf('Mle: EXG(%.3f, %.3f, %.3f)', e.mu, e.sigma, e.tau), ...
    sprintf('Old: EXG(%.3f, %.3f, %.3f)', f.mu, f.sigma, f.tau)});

title('estimated distribution');


% Copyright notice
%    Copyright 2016 Benjamin T. Files
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