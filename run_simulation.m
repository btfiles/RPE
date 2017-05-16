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
%
% To get a sense of the variability in the ML method vs the Regression
% method, I'm going to run the same simulation 100 times.
outfile = 'hrfarsim4-10ms.mat';
s_nsim = 100; % per condition

s_conditions = [.6, .01;
    .6, .001;
    .9, .05;
    .99, .05];

% change to
rng('default')
% if you want to get a sense of how much variability you can get on
% repeated runs.

% stimulation settings
s_stim_rate = 4; % Stim/s
s_block_length = 60; % s
s_inter_block_interval = 10; % s
s_n_block = 10; % number of blocks
s_pTar = 0.1; % proportion of stimuli that are targets

% exgaussian RT parameters
s_mu = 0.3;
s_sigma = 0.1;
s_tau = 0.15;

% time resolution of the whole business
time_res = 0.01; % seconds


%% values derived from settings
% exgaussian random numbers
exgr = @(sz) normrnd(s_mu, s_sigma, sz) + exprnd(s_tau, sz);

% number of conditions to simulate
n_cond = size(s_conditions, 1);


%% Initialize trackers
sim_results = zeros(s_nsim, 6, n_cond); % wHr, wFar, rHr, rFar, mlHr, mlFar
sim_times = zeros(s_nsim, 3, n_cond);

for iCond = 1:n_cond
    
    % True performance parameters
    pHit = s_conditions(iCond, 1); % hit rate
    pFa = s_conditions(iCond, 2); % false alarm rate
    fprintf(1, '============ Starting condition %d: HR: %f, FAR: %f ============\n', ...
        iCond, pHit, pFa);
    %%
    for iSim = 1:s_nsim
        %% Run the simulation
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
        fprintf(1, 'Window analysis %d\n', iSim);
        t0 = tic;
        win_lo = 0.0;
        win_hi = 1.0;
        in_any_win = false(size(button_time));
        n_hit = 0;
        for iTar = 1:numel(tar_times)
            tt = tar_times(iTar);
            in_win = button_time > tt + win_lo & button_time < tt+win_hi;
            if any(in_win),
                n_hit = n_hit + 1;
            end
            in_any_win(in_win) = true;
        end
        
        win_hr = n_hit/numel(tar_times);
        win_far = sum(~in_any_win)/numel(nt_times);
        
        sim_results(iSim, 1:2, iCond) = [win_hr, win_far];
        twin = toc(t0);
        sim_times(iSim, 1, iCond) = twin;
        fprintf(1, 'Finished window in %f s\n', twin);
        
        %% Do the regression estimation
        fprintf(1, 'Regression analysis %d\n', iSim);
        t0 = tic;
        f = rpe.RSVPPerformanceEstimator(stim_time, stim_label, button_time);
        f.time_resolution = time_res;
        [hrr, farr] = f.runEstimates();
        treg = toc(t0);
        
        sim_results(iSim, 3:4, iCond) = [hrr, farr];
        sim_times(iSim, 2, iCond) = treg;
        fprintf(1, 'Finished regression in %f s\n', treg);
        
        %% Do the maximum likelihood estimation
        % setup the estimator
        % clear e
        fprintf(1, 'Maximum Likelihood estimation %d\n', iSim);
        t0 = tic;
        e = rpe.RSVPPerformanceML(stim_time, stim_label, button_time);
        e.time_resolution = time_res;
        
        % Now run the estimator
        [hr, far] = e.runEstimates();
        tml = toc(t0);
        sim_results(iSim, 5:6, iCond) = [hr, far];
        sim_times(iSim, 3, iCond) = tml;
        fprintf(1, 'Finished mle in %f s\n', tml);
        
        % This step might take a few minutes, depending on how many stimuli you
        % give it and how close the stimuli are together in time. Also, responses
        % that are very close together in time are potentially problematic. In this
        % example, a warning is thrown but otherwise ignored.
    end
    
    %%
    figure;
    subplot(1,2,1);
    boxplot(squeeze(sim_results(:, 1:2:end, iCond)), {'win', 'reg', 'mle'});
    ylabel('hr');
    hold on;
    plot(xlim, pHit.*[1 1], 'k--');
    subplot(1,2,2);
    boxplot(squeeze(sim_results(:, 2:2:end, iCond)), {'win', 'reg', 'mle'});
    ylabel('far');
    hold on;
    plot(xlim, pFa.*[1 1], 'k--');
    drawnow
    pause(.2);
end

save(outfile, 'sim_results', 'sim_times', 's_*');
%%
for iCond = 1:size(s_conditions,1)
    pHit = s_conditions(iCond, 1);
    pFa = s_conditions(iCond, 2);
    %%
    figure('name', sprintf('Contrast%dBoxplots', iCond));
    subplot(1,2,1);
    boxplot(squeeze(sim_results(:, 1:2:end, iCond)), {'win', 'reg', 'mle'});
    ylabel('hr');
    hold on;
    plot(xlim, pHit.*[1 1], 'k--');
    subplot(1,2,2);
    boxplot(squeeze(sim_results(:, 2:2:end, iCond)), {'win', 'reg', 'mle'});
    ylabel('far');
    hold on;
    plot(xlim, pFa.*[1 1], 'k--');
    drawnow;
    pause(.2);
end

%% summary of rmse
sz = size(s_conditions, 1);
tmp = zeros(1, size(sim_results,2), sz);
for i = 1:sz, 
    tmp(1, 1:2:end, i) = s_conditions(i, 1); 
    tmp(1, 2:2:end, i) = s_conditions(i, 2); 
end
err = bsxfun(@minus, sim_results, tmp);
rmse = squeeze(sqrt(mean(err.^2, 1)));

err_rs = reshape(err, size(err,1), []);
bsf = @(d) sqrt(mean(d));
tic
ci_rs = bootci(10000, {bsf, err_rs.^2}, 'Options', struct('UserParallel', false));
toc

ci = reshape(ci_rs, 2, size(err,2), size(err,3));
%% Show summary
xtl = sprintf('H: %1.3f,F: %1.3f|', s_conditions');
xtl = strsplit(xtl(1:(end-1)), '|');

L = rmse - squeeze(ci(1, :, :));
U = squeeze(ci(2, :, :)) - rmse;

figure('name', 'overview', 'position', [996   790   570   689]); 
subplot(2,1,1); 
errorbar(repmat(1:sz, size(rmse,1)/2, 1)', rmse(1:2:end, :)', L(1:2:end, :)', U(1:2:end, :)', '.-'); 
ylabel('RMSE');
xlim([0, sz+1]);
set(gca, 'xtick', (1:sz), 'XTickLabel', xtl, 'XTickLabelRotation', 45);
xlabel('condition');
drawnow;
pause(.05);
legend({'win hr', 'reg hr', 'mle hr'}, 'location', 'best'); 
subplot(2,1,2); 
errorbar(repmat(1:sz, size(rmse,1)/2, 1)', rmse(2:2:end, :)', L(2:2:end, :)', U(2:2:end, :)', '.-'); 
ylabel('RMSE');
xlim([0, sz+1]);
set(gca, 'xtick', (1:sz), 'XTickLabel', xtl, 'XTickLabelRotation', 45);
xlabel('condition');
drawnow;
pause(.05);
legend({'win far', 'reg far', 'mle far'}, 'location', 'best');