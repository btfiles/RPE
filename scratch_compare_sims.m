hires_fn = 'hrfarsim4.mat';
lores_fn = 'hrfarsim4-10ms.mat';

hi = load(hires_fn);
lo = load(lores_fn);
%%
bo = [hi, lo];
[rmse, ci] = deal(cell(size(bo)));
for iSim = 1:numel(bo)
    s = bo(iSim);
    
    sz = size(s.sim_results);
    tru = zeros(1, sz(2), sz(3));
    tru(1, 1:2:end, :) = repmat(s.s_conditions(:, 1)', sz(2)/2, 1);
    tru(1, 2:2:end, :) = repmat(s.s_conditions(:, 2)', sz(2)/2, 1);
    err2 = bsxfun(@minus, s.sim_results, tru).^2;
    err2 = reshape(err2, sz(1), []);
    rmse{iSim} = reshape(squeeze(sqrt(mean(err2, 1))), sz(2), sz(3));
    
    
    bsf = @(d) sqrt(mean(d));
    tic
    cil = bootci(10000, {bsf, err2}, 'Options', struct('UserParallel', false));
    toc
    cil = reshape(cil, 2, sz(2), sz(3));
    ci{iSim} = cil;
end

%% We're only interested in mle performance

rmsem = cat(ndims(rmse{1})+1, rmse{:});

rmse_hr = squeeze(rmsem(5, :, :));

cim = cat(ndims(ci{1})+1, ci{:});

hrL = bsxfun(@minus, rmse_hr, squeeze(cim(1, 5, :, :)));
hrU = bsxfun(@minus, squeeze(cim(2, 5, :, :)), rmse_hr);


rmse_far = squeeze(rmsem(6, :, :));
farL = bsxfun(@minus, rmse_far, squeeze(cim(1, 6, :, :)));
farU = bsxfun(@minus, rmse_far, squeeze(cim(1, 6, :, :)));

%% display results
xtl = sprintf('H: %1.3f,F: %1.3f|', lo.s_conditions');
xtl = strsplit(xtl(1:(end-1)), '|');
x = repmat((1:size(rmse_hr,1))', 1, numel(bo));

figure('name', 'ResCompare', 'Position', [996   625   612   854]);
subplot(2,1,1)
errorbar(x, rmse_hr, hrL, hrU)
ylabel('HR RMSE');
xlim([0, sz(end)+1]);
set(gca, 'xtick', (1:sz(end)), 'XTickLabel', '', 'XTickLabelRotation', 45);
xlabel('condition');
drawnow;
pause(.05);
legend({'HR hi-res', 'HR lo-res'}, 'location', 'best'); 

subplot(2,1,2)
errorbar(x, rmse_far, farL, farU)
ylabel('FAR RMSE');
xlim([0, sz(end)+1]);
set(gca, 'xtick', (1:sz(end)), 'XTickLabel', xtl, 'XTickLabelRotation', 45);
xlabel('condition');
drawnow;
pause(.05);
legend({'FAR hi-res', 'FAR lo-res'}, 'location', 'best'); 