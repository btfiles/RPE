function p = pRespLocal(t_stim, rr, t_step, t_support, kmax, epdf, t_min, t_max)
% probability of responding in a local piece of an RSVP experiment.

% t_stim = 0:.1:1.5;
if numel(rr)==1
    rr = rr.*ones(size(t_stim));
end
assert(numel(rr)==numel(t_stim), 'Resonse Rate must be either scalar or the same size as t_stim.');

if kmax > 14
    warning('Selecting a high value of kmax can cause this algorithm to be very slow. Consider kmax < 14.');
end
if numel(t_stim)~=1 && kmax > numel(t_stim)
    warning('Reducing kmax to numel t_stim. N choose K with K>N is undefined.');
    kmax = numel(t_stim);
end

% Create a matrix of individual probabilities
t = 0:t_step:(max(t_stim)+t_support);
idx = 0:(t_support/t_step);
% epdf = rtpdf(t_pdf).*t_step; % This is slow. can we optimize?
ind_prob = zeros(numel(t_stim), numel(t));

start_idx = round((t_stim/t_step)+1);
for iStim = 1:numel(t_stim)
    ind_prob(iStim, start_idx(iStim) + idx) = epdf*rr(iStim);
end

% cull out just the part we care about
cull_start = floor(t_min/t_step)+1;
cull_end = floor(t_max/t_step)+1;

ind_prob = ind_prob(:, cull_start:cull_end);

if numel(t_stim)==1
    % no interactions to consider
    p = ind_prob;
    return;
end

% compute interaction terms of order 2 - kmax.
k_list = 2:kmax;
nprods = arrayfun(@(k) nchoosek(numel(t_stim), k), k_list);
prodc = cell(numel(k_list), 1);
% tic
for i = 1:numel(k_list)
    k = k_list(i);
    sel_mat = nchoosek(1:numel(t_stim), k);
    prods = zeros(nprods(i), size(ind_prob, 2));
    for j = 1:size(sel_mat, 1)
        prods(j, :) = prod(ind_prob(sel_mat(j,:), :), 1); 
    end
    prodc{i} = prods;
end
% toc

% p = sum of individual probabilities minus their co-occurances (i.e. their
% products)
p = sum(ind_prob) - sum(cat(1, prodc{:}));