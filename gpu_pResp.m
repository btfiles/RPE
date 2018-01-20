%% Getting to GPU
% It seems like GPU computation ought to be a lot faster than CPU for the
% RSVP Performance Estimation problem. But I can't seem to get it to work.
%
% Conlusion: can't run this efficiently on the GPU
% Even parallelizing on the CPU seems to slow it down by about 2x.


function p = gpu_pResp(hr, far, exg, stimulus_times, stimulus_labels, ...
    t_init, time_resolution, num_t, pdf_support)

% Computes the probability of responding in each time bin given
% HR and FAR.
% p_all = pResp(obj, hr, far, exg)
%
% Calculates the joint probability of one or more responses
% occurring in each time bin. An object property, kmax, defines
% the highest-order interaction this computation takes into
% account. This determines how many stimuli generating a
% response at a given time bin will be taken into account. If
% kmax is set to 3, for example, then the possibility that any
% combination of 3 (but not 4 or more) of however many nearby
% stimuli might produce a response ocurred simultaneously.
%
% This version is attempting to run efficiently on GPU

pdf = rpe.exGaussPdf(0:time_resolution:pdf_support, exg(1), exg(2), exg(3)).*time_resolution;

% response rate array: HR for targets, FAR for non-targets
rra = far*ones(size(stimulus_labels));
rra(stimulus_labels==true) = hr;

%% put stimuli in the same time bins as the responses.
stim_idx = round((stimulus_times-t_init)/time_resolution) + 1;

%% Ready to compute probabilities
% We want the probability of a response in each response time
% bin. Not all stimuli contribute to the probability at each
% time bin, fortunately, so we get some efficiency by
% identifying those stimuli that do contribute, and only use
% those in computations for that time bin. Also, the set of
% stimuli that do contribute is fixed for the duration between
% consecutive stimuli, so we don't have to iterate over each
% time bin; instead we iterate over each stimulus and do a
% vectorized computation on that time chunk.

% [lp, istart, istop] = arrayfun(@(idx) pChunk(idx, stim_idx, rra, pdf), 1:numel(stim_idx), 'uni', false);
% 
% p = zeros(1, num_t);
% for i = 1:numel(lp)
%     p(istart{i}:istop{i}) = lp{i};
% end
lp = cell(numel(stim_idx), 1);
[istart, istop] = deal(zeros(numel(stim_idx), 1));
for i = 1:numel(stim_idx)
    [lp{i}, istart(i), istop(i)] = pChunk(i, stim_idx, rra, pdf);
end
p = zeros(1, num_t);
for i = 1:numel(lp)
    p(istart(i):istop(i)) = lp{i};
end
end

function [p, idx_start, idx_stop] = pChunk(idx, st, rra, pdf)

pdfl = length(pdf);

idx_start = st(idx)+1;

if idx==numel(st)
    idx_stop = idx_start+pdfl;
else
    idx_stop = min(idx_start+pdfl, st(idx+1));
end

candstimidx = (st>(idx_start-pdfl) & st<=idx_start);
if ~any(candstimidx)
    p = zeros(1, (idx_stop-idx_start));
    return;
end

cstims = st(candstimidx);
rr = rra(candstimidx);
rowp = zeros(numel(cstims), idx_stop-idx_start+1);
for i = 1:numel(cstims)
    sprev = cstims(i);
    scurr = st(idx);
    snext = idx_stop;
    pdfs = scurr-sprev+1;
    pdfe = min(snext-sprev, pdfl);
    psnip = pdf(pdfs:pdfe).*rr(i);
    rowp(i, 1:numel(psnip)) = psnip;
end
p = 1-prod(1-rowp);

end