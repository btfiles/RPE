function [win_hr, win_far, reg_hr, reg_far, reg_rt, ml_hr, ml_far, ml_rt] = ...
    estimateRsvpPerformance(button_times, stim_time, stim_label)
% Run three different estimators of RSVP performance.
% [win_hr, win_far, reg_hr, reg_far, reg_rt, ml_hr, ml_far, ml_rt] = estimateRsvpPerformance(button_times, stim_time, stim_label)
%
% Input arguments
% button_time - time (in seconds) at which the button was pressed
% stim_time   - time (in seconds) of stimulus onsets
% stim_label  - 0 for non-targets, 1 for targets
%
% Outputs:
% win_hr, win_far - hit rate and false alarm rate for the window method
% reg_hr, reg_far, reg_rt - hit rate, false alarm rate and RT distribution
%                           parameter estimates for regression method.
% ml_hr, ml_far, ml_rt - hit rate, false alarm rate and RT distribution
%                        parameter estimates for MLE method.

% Window method parameters
win_lo = 0;
win_hi = 1;

% time resolution is shared
s_time_res = 0.01;

%% Convert to tar_times, nt_times
tar_times = stim_time(stim_label==1);
nt_times = stim_time(stim_label==0);

%% Conventional window analysis
% The idea of the window analysis is that any response that falls
% within a window of time after a target is a hit, while all others are
% a false alarm. It gets a little complicated when multiple responses
% fall within a window and/or the windows of multiple target stimuli
% overlap.

fprintf(1, 'Window estimate...\n');
t0 = tic;
claimed_hit = false(size(button_times)); % track if a response has been claimed by a target image

for iTar = 1:numel(tar_times)
    tt = tar_times(iTar);
    
    % find responses that fall within the window of time following this
    % target
    in_win = button_times > tt + win_lo & button_times < tt+win_hi;
    
    % What if there are multiple responses in the window?
    % Claim the first one that has not previously been claimed. Leave
    % the rest as either false alarms or hits to later targets.
    if nnz(in_win & ~claimed_hit)>1
        in_win_idx = find(in_win & ~claimed_hit);
        in_win(in_win_idx(2:end)) = false;
    end
    if any(in_win & ~claimed_hit)
        claimed_hit(in_win & ~claimed_hit) = true;
    end
    
end

win_hr = sum(claimed_hit)/numel(tar_times);
win_far = sum(~claimed_hit)/numel(nt_times);

twin = toc(t0);
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

fprintf(1, 'Regression analysis...\n');
t0 = tic;
reg = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_times);
reg.time_resolution = s_time_res; % for non-default time res
[reg_hr, reg_far] = reg.runEstimates();
treg = toc(t0);

fprintf(1, 'Finished regression in %f s\n', treg);

reg_rt = [reg.mu, reg.sigma, reg.tau];

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

%% Do the maximum likelihood estimation 2
% This is similar to the maximum likelihood estimator above, but
% instead of estimating the parameters of the rt distribution
% separately, we estimate them simultaneously with the hr and far.

fprintf(1, 'Maximum Likelihood simultaneous estimation...\n');
t0 = tic;
e2 = rpe.RSVPPerformanceML2(stim_time, stim_label, button_times);
e2.time_resolution = s_time_res;

% Now run the estimator
[ml_hr, ml_far] = e2.runEstimates();
tml2 = toc(t0);
fprintf(1, 'Finished simultaneous mle in %f s\n', tml2);
ml_rt = [e2.mu, e2.sigma, e2.tau];