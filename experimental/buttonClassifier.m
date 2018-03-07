function scores = buttonClassifier(stim_times, resp_times, pdf, pdf_support)

scores = cell(numel(resp_times), 1);
parfor idx = 1:numel(resp_times)
    rt = resp_times(idx);
    st_idx = stim_times>=rt-pdf_support & stim_times < rt;
    st = rt-stim_times(st_idx);
    
    % special cases for weird numbers of stim times
    if numel(st) == 0
        continue
    end
    
    % likelihood
    lik = pdf(st); %#ok
    
    % scaled likelihood
    lik = lik./(sum(lik));
    sc = zeros(1, numel(stim_times));
    sc(st_idx) = lik;
    scores{idx} = sc;
end

scores = sum(cat(1, scores{:}));