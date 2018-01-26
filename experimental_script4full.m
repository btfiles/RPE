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
%rng('shuffle')
% if you want to get a sense of how much variability you can get on
% repeated runs.

rngSeed = 37;
rng(rngSeed) 

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

 conditions = ...
     [.5 .1 mu sigma tau;
     .5 .05 mu sigma tau;
     .5 .01 mu sigma tau;
     .5 .005 mu sigma tau;
     .5 .001 mu sigma tau;
     .75 .1 mu sigma tau;
     .75 .05 mu sigma tau;
     .75 .01 mu sigma tau;
     .75 .005 mu sigma tau;
     .75 .001 mu sigma tau;
     .9 .1 mu sigma tau;
     .9 .05 mu sigma tau;
     .9 .01 mu sigma tau;
     .9 .005 mu sigma tau;
     .9 .001 mu sigma tau;
     .95 .1 mu sigma tau;
     .95 .05 mu sigma tau;
     .95 .01 mu sigma tau;
     .95 .005 mu sigma tau;
     .95 .001 mu sigma tau;
     1.0 .1 mu sigma tau;
     1.0 .05 mu sigma tau;
     1.0 .01 mu sigma tau;
     1.0 .005 mu sigma tau;
     1.0 .001 mu sigma tau;];

map_estimates = zeros([size(conditions), nsim]);
reg_estimates = zeros([size(conditions), nsim]);
win_estimates = zeros([size(conditions), nsim]);
[ll_true, ll_estimate, reg_t, ml_t, win_t, converged] = deal(zeros(size(conditions,1), nsim));
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
    
    runName = sprintf('%.2f_%.3f_%.2f_%.2f_%.2f',pHit,pFa,mu,sigma,tau);
    status = sprintf('Running %s. Condition # %d.',runName,iCond); disp(status);
    
    % If we've already started this condition combo, load the previous
    % results from the saved workspace
    if exist(['workspaces/' runName '.mat'], 'file') == 2
        load(['workspaces/' runName '.mat']);
    end
    
    for iSim = 1:nsim
        % If we've already done this iteration of this condition combo, skip it
        if ml_t(iCond, iSim) ~= 0
            fprintf(1, 'Skipping iteration %d of condition %d.', iSim, iCond);
            continue;
        end
        
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
        
        %% Conventional window analysis
        % The idea of the window analysis is that any response that falls
        % within a window of time after a target is a hit, while all others are
        % a false alarm. It gets a little complicated when multiple responses
        % fall within a window and/or the windows of multiple target stimuli
        % overlap.

        fprintf(1, 'Window analysis %d\n', iSim);
        tw = tic;
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
        win_estimates(iCond, :, iSim) = [win_hr, win_far, mu, sigma, tau];
        win_t(iCond, iSim) = toc(tw);
        fprintf(1, 'Finished window in %f s\n', win_t(iCond, iSim));
        
        %% Run the regression estimator
        tr = tic;
        e = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
        e.time_resolution = 0.01;
        [hr, far] = e.runEstimates();
        reg_t(iCond, iSim) = toc(tr);
        
        reg_estimates(iCond, :, iSim) = [hr, far, e.mu, e.sigma, e.tau];
        fprintf(1, 'Finished regression in %f s\n', reg_t(iCond, iSim));
        
        %% Run the MAP estimator
        tm = tic;
        m = rpe.RSVPPerformanceMAP(stim_time, stim_label, button_time);
        m.time_resolution = 0.01;
        m.use_prior = false;
        m.diag = false;
        
        nattempt = 0;
        exit_flag = 0;
        while (nattempt < MAX_ATTEMPT && exit_flag ~= 1)
            [hr_m, far_m, exit_flag] = m.estimatePerformance();
            nattempt = nattempt+1;
        end
        if exit_flag ~= 1
            warning('After %d attempts, no convergence occured.', nattempt);
        end
        converged(iCond, iSim) = exit_flag;
        ml_t(iCond, iSim) = toc(tm);
        map_estimates(iCond, :, iSim) = [hr_m, far_m, m.mu, m.sigma, m.tau];
        fprintf(1, 'Finished MLE in %f s (%d attempts)\n', ml_t(iCond, iSim), nattempt);
        
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
        %waitbar(simcount/(size(conditions,1)*nsim), wb);
        
        %Save data
        save(['workspaces/' runName '.mat']);
    end
end
totalTime = toc(t0)