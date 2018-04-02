%% Overview
% This script shows an example of how to use the RSVPPerformanceBayes.
%
% Note, this is for testing more than demonstration, as it is usually
% preferable to use the much faster RSVPPerformanceMLE. In fact, using
% RSVPPerformanceBayes directly is almost certainly a bad idea.
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
% then initialize the estimator like so: 
%
% m = rpe.RSVPPerformanceBayes(stim_time, stim_label, button_time);
%
% and run the estimator: 
%
% [hr, far] = m.estimatePerformance; 
%
% This will take a while; on a fast, 24 core machine it would take most of
% a day to do this with any reasonable chain length. This uses a custom
% Metropolis-within-Gibbs sampler rpe.MWG, which is not particularly fast.
% The likelihood computation is slow, but if it could be encoded in e.g.
% Stan, the compiled version would probably operate much faster.
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
% rng('default'); 
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
pFa = 0.1; % false alarm rate

% exgaussian RT parameters
mu = 0.3;
sigma = 0.1;
tau = 0.15;

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
%% Run the Bayesian estimate:
m = rpe.RSVPPerformanceBayes(stim_time, stim_label, button_time);
m.time_resolution = 0.1;

m.samp_per_chain = 100;
m.warmup_per_chain = 100;
m.batch_update = 20;

[hr, far] = m.estimatePerformance();

%% Print out the results
fprintf(1, 'Estimated HR: %.4f simulated HR: %.4f\n', hr, pHit);
fprintf(1, 'Estimated FAR: %.4f simulated FAR: %.4f\n', far, pFa);

%% 
% Copyright notice
%    Copyright 2018 Benjamin T. Files
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