rng('default');
%% general settings
p_tar = .1;
t_support = 1.5;
t_step = 0.001;
mu = .3;
sig = .1;
tau = .2;

rtpdf = @(x) exgaussPdf(x, mu, sig, tau);
t_pdf = 0:t_step:t_support;

rr = 1; % response rate

%% setup a mini-example
t_stim = 0:.2:(8*60);
kmax = 3;
hr = .4;
far = .01;
labels = false(size(t_stim));
labels(randi(numel(labels), 1, floor(numel(labels)*p_tar))) = true;

rra = far.*ones(size(labels));
rra(labels) = hr;


%% run the example
t0 = tic;
p = pResp(t_stim, rra, rtpdf, t_step, t_support, kmax); 
fprintf(1, 'Function took %f s\n', toc(t0));

%% visualize
t = 0:t_step:(max(t_stim)+t_support);
figure;
plot(t, p);
hold on;
stem(t_stim, labels.*max(p), 'marker', '.');
return
%% First optimization 
% Once t_stim starts to exceed t_support by a lot (and it will), the above
% ends up doing a lot of multiplications by zero. We can avoid computing
% the full ind_prob matrix and avoid that.
ind_prob = zeros(numel(t_stim), numel(t));
for iStim = 1:numel(t_stim)
    start_idx = round((t_stim(iStim)/t_step)+1);
    ind_prob(iStim, start_idx + (0:(t_support/t_step))) = rtpdf(t_pdf).*t_step*rra(iStim);
end

%%
t = 0:t_step:(max(t_stim)+t_support);
stim_idx = round(t_stim/t_step) + 1;

p_all = zeros(size(t));
t0 = tic;
for iStim = 1:numel(t_stim)
    
    in_hood = (t_stim >= (t_stim(iStim) - t_support)) & ...
        (t_stim <= t_stim(iStim));
    
    hood_idx = find(in_hood);
    t_stim_local = t_stim(hood_idx) - t_stim(hood_idx(1)); % set first one to zero

    
    % This is an inefficiency, we're computing time points we're going to
    % discard.
    p_local = pResp(t_stim_local, rra(in_hood), rtpdf, t_step, t_support, kmax);
    
    % and now we need to figure out what part to keep
    if iStim == numel(t_stim)
        t_next = t_stim(end)+t_support;
    else
        t_next = t_stim(iStim+1);
    end
    
    t_next_local = t_next - t_stim(hood_idx(1));
    t_curr_local = t_stim_local(end);
    
    local_idx = (floor(t_curr_local/t_step) + 1):(floor(t_next_local/t_step)+1);
    
    p_all(stim_idx(hood_idx(1)) - 1 + local_idx) = p_local(local_idx);
end
    
fprintf(1, 'Optimized took %f s\n', toc(t0));

%%
figure;
plot(t, p_all);
hold on;
stem(t_stim, labels.*max(p_all), 'marker', '.');