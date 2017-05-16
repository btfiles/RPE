function p_all = pResp(t_stim, rra, rtpdf, t_step, t_support, kmax)
%%
t = 0:t_step:(max(t_stim)+t_support);
stim_idx = round(t_stim/t_step) + 1;

t_pdf = 0:t_step:t_support;
epdf = rtpdf(t_pdf).*t_step;

p_all = zeros(size(t));
for iStim = 1:numel(t_stim)
    
    in_hood = (t_stim >= (t_stim(iStim) - t_support)) & ...
        (t_stim <= t_stim(iStim));
    
    hood_idx = find(in_hood);
    t_stim_local = t_stim(hood_idx) - t_stim(hood_idx(1)); % set first one to zero

    % and now we need to figure out what part to keep
    if iStim == numel(t_stim)
        t_next = t_stim(end)+t_support;
    else
        t_next = t_stim(iStim+1);
    end
    
    t_min = t_stim(iStim) - t_stim(hood_idx(1));
    t_max = t_next - t_stim(hood_idx(1));
    p_local = pRespLocal(t_stim_local, rra(in_hood), t_step, t_support, kmax, epdf, t_min, t_max);
    p_all(stim_idx(iStim) - 1 + (1:numel(p_local))) = p_local;
end
    