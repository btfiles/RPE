% this script illustrates how to use the function estimateRsvpPerformance.
example_setfile = '/home/bfiles/matlab/CompressedRSVP/data/S01_eyeMovementsRemoved_0p1Hz.set';

%%
eeglab('nogui');
eeg = pop_loadset(example_setfile);

%% Event key:
% 1 - nontarget
% 2 - target
% 3 - buttonpress

% let's extract a subset; block 2
evt = eeg.event([eeg.event.block]==2);

% identify stimulus events
is_stim = [evt.type] == 1 | [evt.type] == 2;
fprintf(1, 'Found %d stimulus events.\n', sum(is_stim));

% identify stimulus latencies and labels
stim_latency = [evt(is_stim).latency];
stim_label = zeros(size(stim_latency));
stim_label([evt(is_stim).is_tar]==1) = 1;

% identify button press events
is_bp = [evt.type]==3;
fprintf(1, 'Found %d button press events.\n', sum(is_bp));
button_latency = [evt(is_bp).latency];

% convert to time
stimulus_time = stim_latency./eeg.srate;
button_time = button_latency./eeg.srate;

%% Run the estimators

[win_hr, win_far, reg_hr, reg_far, reg_rt, ml_hr, ml_far, ml_rt] = ...
    rpe.estimateRsvpPerformance(button_time, stimulus_time, stim_label);

%% Display results:
fprintf(1, 'Window method\n\tHR: %.3f, FAR: %.3f\n', win_hr, win_far);
fprintf(1, 'Regression method\n\tHR: %.3f, FAR: %.3f, mu: %.3f, sigma: %.3f, tau: %.3f\n', reg_hr, reg_far, reg_rt);
fprintf(1, 'MaxLik method\n\tHR: %.3f, FAR: %.3f, mu: %.3f, sigma: %.3f, tau: %.3f\n', ml_hr, ml_far, ml_rt);
